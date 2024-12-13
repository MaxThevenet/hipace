/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "HelmholtzDeposition.H"
#include "DepositionUtil.H"
#include "particles/beam/BeamParticleContainer.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "helmholtz/Helmholtz.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"
#include "Hipace.H"

void
HelmholtzDeposition (BeamParticleContainer& beam, Helmholtz& helmholtz,
                    const bool do_dtau,
                    const int which_beam_slice,
                    const int islice,
                    const int isubslice)
{
    HIPACE_PROFILE("HelmholtzDeposition()");

    using namespace amrex::literals;

    amrex::FArrayBox& isl_fab = helmholtz.getSlices()[0];
    const amrex::Geometry& gm = helmholtz.GetHelmholtzGeom();
    const CheckDomainBounds helmholtz_bounds {gm};
    const int rev_sub = helmholtz.GetNSubSlices() - isubslice;
    const amrex::Real sub_frac_lo = amrex::Real{rev_sub-1} / helmholtz.GetNSubSlices();
    const amrex::Real sub_frac_hi = amrex::Real{rev_sub} / helmholtz.GetNSubSlices();
    const amrex::Real min_z = gm.ProbLo(2) +
        (islice-gm.Domain().smallEnd(2)+sub_frac_lo)*gm.CellSize(2);
    const amrex::Real max_z = gm.ProbLo(2) +
        (islice-gm.Domain().smallEnd(2)+sub_frac_hi)*gm.CellSize(2);

    // Offset for converting positions to indexes
    amrex::Real const x_pos_offset = GetPosOffset(0, gm, gm.Domain());
    amrex::Real const y_pos_offset = GetPosOffset(1, gm, gm.Domain());

    PhysConst const phys_const = get_phys_const();

    // Extract box properties
    const amrex::Real dxi = gm.InvCellSize(0);
    const amrex::Real dyi = gm.InvCellSize(1);
    const amrex::Real dzi = gm.InvCellSize(2) * helmholtz.GetNSubSlices();
    amrex::Real invvol = dxi * dyi * dzi;

    if (Hipace::m_normalized_units) {
        const amrex::Geometry& lev0_geom = Hipace::GetInstance().m_3D_geom[0];
        invvol = lev0_geom.CellSize(0) * lev0_geom.CellSize(1) * dxi * dyi;
    }

    const amrex::Real clightinv = 1.0_rt/(phys_const.c);
    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);
    const amrex::Real q = beam.m_charge;

    amrex::AnyCTO(
        // use compile-time options
        amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
        {Hipace::m_depos_order_xy},
        // call deposition function
        // The three functions passed as arguments to this lambda
        // are defined below as the next arguments.
        [&](auto is_valid, auto get_cell, auto deposit){
            constexpr auto ctos = deposit.GetOptions();
            constexpr int depos_order = ctos[0];
            constexpr int stencil_size = depos_order + 1;
            SharedMemoryDeposition<stencil_size, stencil_size, true>(
                beam.getNumParticles(which_beam_slice), is_valid, get_cell, deposit,
                isl_fab.array(), isl_fab.box(),
                beam.getBeamSlice(which_beam_slice).getParticleTileData(),
                std::array<int, 0>{},
                std::array{
                    do_dtau ? -1 : WhichHelmholtzSlice::jx_n00jm1,
                    do_dtau ? -1 : WhichHelmholtzSlice::jy_n00jm1,
                    do_dtau ? -1 : WhichHelmholtzSlice::jz_n00jm1,
                    do_dtau ? -1 : WhichHelmholtzSlice::rho_n00jm1,
                    do_dtau ? WhichHelmholtzSlice::dtau_jx_n00j00 : -1,
                    do_dtau ? WhichHelmholtzSlice::dtau_jy_n00j00 : -1,
                    do_dtau ? WhichHelmholtzSlice::dtau_jz_n00j00 : -1
                });
        },
        // is_valid
        // return whether the particle is valid and should deposit
        [=] AMREX_GPU_DEVICE (int ip, auto ptd, auto /*depos_order*/)
        {
            const amrex::Real xp = ptd.pos(0, ip);
            const amrex::Real yp = ptd.pos(1, ip);
            const amrex::Real zp = ptd.pos(2, ip);
            return ptd.id(ip).is_valid() &&
                   helmholtz_bounds.contains(xp, yp) &&
                   zp >= min_z &&
                   zp < max_z;
        },
        // get_cell
        // return the lowest cell index that the particle deposits into
        [=] AMREX_GPU_DEVICE (int ip, auto ptd, auto depos_order) -> amrex::IntVectND<2>
        {
            // --- Compute shape factors
            // x direction
            const amrex::Real xmid = (ptd.pos(0, ip) - x_pos_offset)*dxi;
            // i_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            amrex::Real sx_cell[depos_order + 1];
            const int i = compute_shape_factor<depos_order>(sx_cell, xmid);

            // y direction
            const amrex::Real ymid = (ptd.pos(1, ip) - y_pos_offset)*dyi;
            amrex::Real sy_cell[depos_order + 1];
            const int j = compute_shape_factor<depos_order>(sy_cell, ymid);

            return {i, j};
        },
        // deposit
        // deposit the charge / current of one particle
        [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                              Array3<amrex::Real> arr,
                              auto /*cache_idx*/, auto depos_idx,
                              auto depos_order) {
            // --- Get particle quantities
            const amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip];
            const amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip];
            const amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip];
            const amrex::Real pux = ptd.m_runtime_rdata[0][ip];
            const amrex::Real puy = ptd.m_runtime_rdata[1][ip];
            const amrex::Real puz = ptd.m_runtime_rdata[2][ip];

            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + ux*ux*clightsq
                                                         + uy*uy*clightsq
                                                         + uz*uz*clightsq);
            const amrex::Real pgaminv = 1.0_rt/std::sqrt(1.0_rt + pux*pux*clightsq
                                                         + puy*puy*clightsq
                                                         + puz*puz*clightsq);
            const amrex::Real wq = q*ptd.rdata(BeamIdx::w)[ip]*invvol;

            // wqx, wqy wqz are particle current in each direction
            const amrex::Real wqx = wq*ux*gaminv;
            const amrex::Real wqy = wq*uy*gaminv;
            const amrex::Real wqz = wq*uz*gaminv;
            const amrex::Real wqrho = wq;

            const amrex::Real pwqx = wq*pux*pgaminv;
            const amrex::Real pwqy = wq*puy*pgaminv;
            const amrex::Real pwqz = wq*puz*pgaminv;

            // --- Compute shape factors
            // x direction
            const amrex::Real xmid = (ptd.pos(0, ip) - x_pos_offset)*dxi;
            // i_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            amrex::Real sx_cell[depos_order + 1];
            const int i_cell = compute_shape_factor<depos_order>(sx_cell, xmid);

            // y direction
            const amrex::Real ymid = (ptd.pos(1, ip) - y_pos_offset)*dyi;
            amrex::Real sy_cell[depos_order + 1];
            const int j_cell = compute_shape_factor<depos_order>(sy_cell, ymid);

            // Deposit current into jx, jy, jz, rhomjz
            for (int iy=0; iy<=depos_order; iy++){
                for (int ix=0; ix<=depos_order; ix++){
                    if (!do_dtau) {
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i_cell+ix, j_cell+iy, depos_idx[0]),
                            sx_cell[ix]*sy_cell[iy]*wqx);
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i_cell+ix, j_cell+iy, depos_idx[1]),
                            sx_cell[ix]*sy_cell[iy]*wqy);
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i_cell+ix, j_cell+iy, depos_idx[2]),
                            sx_cell[ix]*sy_cell[iy]*wqz);
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i_cell+ix, j_cell+iy, depos_idx[3]),
                            sx_cell[ix]*sy_cell[iy]*wqrho);
                    } else {
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i_cell+ix, j_cell+iy, depos_idx[4]),
                            sx_cell[ix]*sy_cell[iy]*(wqx-pwqx));
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i_cell+ix, j_cell+iy, depos_idx[5]),
                            sx_cell[ix]*sy_cell[iy]*(wqy-pwqy));
                        amrex::Gpu::Atomic::Add(
                            arr.ptr(i_cell+ix, j_cell+iy, depos_idx[6]),
                            sx_cell[ix]*sy_cell[iy]*(wqz-pwqz));
                    }
                }
            }
        });
}
