/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Severin Diederichs, Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#include "PlasmaParticleContainer.H"
#include "utils/Constants.H"
#include "particles/particles_utils/ParticleUtil.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/IonizationEnergiesTable.H"
#include <cmath>

void
PlasmaParticleContainer::
InitParticles (const amrex::IntVect& a_num_particles_per_cell,
               const amrex::RealVect& a_u_std,
               const amrex::RealVect& a_u_mean,
               const amrex::Real a_radius,
               const amrex::Real a_hollow_core_radius)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InitParticles");
    using namespace amrex::literals;
    clearParticles();
    const int lev = 0;
    const auto dx = ParticleGeom(lev).CellSizeArray();
    const auto plo = ParticleGeom(lev).ProbLoArray();
    const amrex::RealBox a_bounds = ParticleGeom(lev).ProbDomain();

    const int depos_order_1 = Hipace::m_depos_order_xy + 1;
    const bool outer_depos_loop = Hipace::m_outer_depos_loop;

    const int num_ppc = AMREX_D_TERM( a_num_particles_per_cell[0],
                                      *a_num_particles_per_cell[1],
                                      *a_num_particles_per_cell[2]);
    amrex::Real scale_fac = Hipace::m_normalized_units ? 1./num_ppc : dx[0]*dx[1]*dx[2]/num_ppc;

    amrex::IntVect box_nodal{amrex::IndexType::CELL,amrex::IndexType::CELL,amrex::IndexType::CELL};
    amrex::IntVect box_grow{0, 0, 0};
    amrex::Real x_offset = 0._rt;
    amrex::Real y_offset = 0._rt;

    if (ParticleGeom(lev).Domain().length(0) % 2 == 1 && a_num_particles_per_cell[0] % 2 == 1) {
        box_nodal[0] = amrex::IndexType::NODE;
        box_grow[0] = -1;
        x_offset = -0.5_rt;
    }

    if (ParticleGeom(lev).Domain().length(1) % 2 == 1 && a_num_particles_per_cell[1] % 2 == 1) {
        box_nodal[1] = amrex::IndexType::NODE;
        box_grow[1] = -1;
        y_offset = -0.5_rt;
    }

    for(amrex::MFIter mfi = MakeMFIter(lev, DfltMfi); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tile_box  = mfi.tilebox(box_nodal, box_grow);

        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        UpdateDensityFunction();
        auto density_func = m_density_func;
        auto density_func_xy = m_density_func_xy;
        auto density_func_z = m_density_func_z;
        const amrex::Real c_light = get_phys_const().c;
        const amrex::Real c_t = c_light * Hipace::m_physical_time;
        const amrex::Real min_density = m_min_density;

        // Count the total number of particles so only one resize is needed
        amrex::Long total_num_particles = amrex::Reduce::Sum<amrex::Long>(tile_box.numPts(),
            [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
            {
                auto [i,j,k] = tile_box.atOffset3d(idx).arr;

                amrex::Long num_particles_cell = 0;
                for (int i_part=0; i_part<num_ppc; ++i_part)
                {
                    amrex::Real r[3];
                    ParticleUtil::get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                    amrex::Real x = plo[0] + (i + r[0] + x_offset)*dx[0];
                    amrex::Real y = plo[1] + (j + r[1] + y_offset)*dx[1];

                    const amrex::Real rsq = x*x + y*y;
                    if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                        y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                        rsq > a_radius*a_radius ||
                        rsq < a_hollow_core_radius*a_hollow_core_radius ||
                        density_func_xy(x, y) * density_func_z(c_t) <= min_density) continue;
//                        density_func(x, y, c_t) <= min_density) continue;

                    num_particles_cell += 1;
                }
                return num_particles_cell;
            });

        if (m_do_symmetrize) {
            total_num_particles *= 4;
            scale_fac /= 4.;
        }

        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        auto old_size = particle_tile.GetArrayOfStructs().size();
        const auto new_size = old_size + total_num_particles;
        particle_tile.resize(new_size);

        const int pid = ParticleType::NextID();
        ParticleType::NextID(pid + total_num_particles);

        ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();
        auto arrdata = particle_tile.GetStructOfArrays().realarray();
        auto int_arrdata = particle_tile.GetStructOfArrays().intarray();
        const int init_ion_lev = m_init_ion_lev;

        // The loop over particles is outside the loop over cells
        // so that particles in the same cell are far apart.
        // This makes current deposition faster.
        for (int i_part=0; i_part<num_ppc; ++i_part)
        {
            amrex::Gpu::DeviceVector<unsigned int> counts(tile_box.numPts(), 0);
            unsigned int* pcount = counts.dataPtr();

            amrex::Gpu::DeviceVector<unsigned int> offsets(tile_box.numPts());
            unsigned int* poffset = offsets.dataPtr();

            amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::Real r[3];

                ParticleUtil::get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                amrex::Real x = plo[0] + (i + r[0] + x_offset)*dx[0];
                amrex::Real y = plo[1] + (j + r[1] + y_offset)*dx[1];

                const amrex::Real rsq = x*x + y*y;
                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    rsq > a_radius*a_radius ||
                    rsq < a_hollow_core_radius*a_hollow_core_radius ||
                    density_func_xy(x, y)*density_func_z(c_t) <= min_density) return;
//                    density_func(x, y, c_t) <= min_density) return;

                int ix = i - lo.x;
                int iy = j - lo.y;
                int iz = k - lo.z;
                int nx = hi.x-lo.x+1;
                int ny = hi.y-lo.y+1;
                int nz = hi.z-lo.z+1;
                unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
                unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
                unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));

                unsigned int cellid = 0;
                if (outer_depos_loop) {
                    // ordering of axes from fastest to slowest:
                    // x/depos_order_1 to match deposition
                    // x%depos_order_1
                    // y
                    // z (not used)
                    // ppc
                    cellid = (uiz * ny + uiy) * nx +
                    uix/depos_order_1 + ((uix%depos_order_1)*nx+depos_order_1-1)/depos_order_1;
                } else {
                    // ordering of axes from fastest to slowest:
                    // x
                    // y
                    // z (not used)
                    // ppc
                    cellid = (uiz * ny + uiy) * nx + uix;
                }

                pcount[cellid] = 1;
            });

            unsigned int num_to_add =
                amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offsets.data());

            if (num_to_add == 0) continue;

            amrex::ParallelForRNG(tile_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, const amrex::RandomEngine& engine) noexcept
            {
                int ix = i - lo.x;
                int iy = j - lo.y;
                int iz = k - lo.z;
                int nx = hi.x-lo.x+1;
                int ny = hi.y-lo.y+1;
                int nz = hi.z-lo.z+1;
                unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
                unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
                unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));

                unsigned int cellid = 0;
                if (outer_depos_loop) {
                    cellid = (uiz * ny + uiy) * nx +
                    uix/depos_order_1 + ((uix%depos_order_1)*nx+depos_order_1-1)/depos_order_1;
                } else {
                    cellid = (uiz * ny + uiy) * nx + uix;
                }

                const amrex::Long pidx = poffset[cellid] - poffset[0] + old_size;

                amrex::Real r[3] = {0.,0.,0.};

                ParticleUtil::get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                amrex::Real x = plo[0] + (i + r[0] + x_offset)*dx[0];
                amrex::Real y = plo[1] + (j + r[1] + y_offset)*dx[1];
                amrex::Real z = plo[2] + (k + r[2])*dx[2];

                const amrex::Real rsq = x*x + y*y;
                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    rsq > a_radius*a_radius ||
                    rsq < a_hollow_core_radius*a_hollow_core_radius ||
                    density_func_xy(x, y)*density_func_z(c_t) <= min_density) return;
//                    density_func(x, y, c_t) <= min_density) return;

                amrex::Real u[3] = {0.,0.,0.};
                ParticleUtil::get_gaussian_random_momentum(u, a_u_mean, a_u_std, engine);

                ParticleType& p = pstruct[pidx];
                p.id()   = pid + int(pidx);
                p.cpu()  = 0; // level 0
                p.pos(0) = x;
                p.pos(1) = y;
                p.pos(2) = z;

//                arrdata[PlasmaIdx::w        ][pidx] = scale_fac * density_func(x, y, c_t);
                arrdata[PlasmaIdx::w        ][pidx] = scale_fac * density_func_xy(x, y) * density_func_z(c_t);
                arrdata[PlasmaIdx::ux       ][pidx] = u[0] * c_light;
                arrdata[PlasmaIdx::uy       ][pidx] = u[1] * c_light;
                arrdata[PlasmaIdx::psi][pidx] = std::sqrt(1._rt+u[0]*u[0]+u[1]*u[1]+u[2]*u[2])-u[2];
                arrdata[PlasmaIdx::x_prev   ][pidx] = x;
                arrdata[PlasmaIdx::y_prev   ][pidx] = y;
                arrdata[PlasmaIdx::ux_half_step][pidx] = u[0] * c_light;
                arrdata[PlasmaIdx::uy_half_step][pidx] = u[1] * c_light;
                arrdata[PlasmaIdx::psi_half_step][pidx] = arrdata[PlasmaIdx::psi][pidx];
#ifdef HIPACE_USE_AB5_PUSH
#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                for (int iforce = PlasmaIdx::Fx1; iforce <= PlasmaIdx::Fpsi5; ++iforce) {
                    arrdata[iforce][pidx] = 0._rt;
                }
#endif
                int_arrdata[PlasmaIdx::ion_lev][pidx] = init_ion_lev;
            });

            old_size += num_to_add;
        }
        if (m_do_symmetrize) {

            const amrex::Real x_mid2 = (ParticleGeom(lev).ProbLo(0) + ParticleGeom(lev).ProbHi(0));
            const amrex::Real y_mid2 = (ParticleGeom(lev).ProbLo(1) + ParticleGeom(lev).ProbHi(1));
            const amrex::Long mirror_offset = total_num_particles/4;
            amrex::ParallelFor(mirror_offset,
            [=] AMREX_GPU_DEVICE (amrex::Long pidx) noexcept
            {
                ParticleType& p = pstruct[pidx];
                const amrex::Real x = p.pos(0);
                const amrex::Real y = p.pos(1);
                const amrex::Real x_mirror = x_mid2 - x;
                const amrex::Real y_mirror = y_mid2 - y;

                const amrex::Real x_arr[3] = {x_mirror, x, x_mirror};
                const amrex::Real y_arr[3] = {y, y_mirror, y_mirror};
                const amrex::Real ux_arr[3] = {-1._rt, 1._rt, -1._rt};
                const amrex::Real uy_arr[3] = {1._rt, -1._rt, -1._rt};

#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                for (int imirror=0; imirror<3; ++imirror) {
                    const amrex::Long midx = (imirror+1)*mirror_offset +pidx;
                    pstruct[midx] = p;
                    pstruct[midx].id() = pid + int(midx);
                    pstruct[midx].pos(0) = x_arr[imirror];
                    pstruct[midx].pos(1) = y_arr[imirror];

                    arrdata[PlasmaIdx::w][midx] = arrdata[PlasmaIdx::w][pidx];
                    arrdata[PlasmaIdx::ux][midx] = arrdata[PlasmaIdx::ux][pidx] * ux_arr[imirror];
                    arrdata[PlasmaIdx::uy][midx] = arrdata[PlasmaIdx::uy][pidx] * uy_arr[imirror];
                    arrdata[PlasmaIdx::psi][midx] = arrdata[PlasmaIdx::psi][pidx];
                    arrdata[PlasmaIdx::x_prev][midx] = x_arr[imirror];
                    arrdata[PlasmaIdx::y_prev][midx] = y_arr[imirror];
                    arrdata[PlasmaIdx::ux_half_step][midx] =
                        arrdata[PlasmaIdx::ux_half_step][pidx] * ux_arr[imirror];
                    arrdata[PlasmaIdx::uy_half_step][midx] =
                        arrdata[PlasmaIdx::uy_half_step][pidx] * uy_arr[imirror];
                    arrdata[PlasmaIdx::psi_half_step][midx] =
                        arrdata[PlasmaIdx::psi_half_step][pidx];
#ifdef HIPACE_USE_AB5_PUSH
#ifdef AMREX_USE_GPU
#pragma unroll
#endif
                    for (int iforce = PlasmaIdx::Fx1; iforce <= PlasmaIdx::Fpsi5; ++iforce) {
                        arrdata[iforce][midx] = 0._rt;
                    }
#endif
                    int_arrdata[PlasmaIdx::ion_lev][midx] = int_arrdata[PlasmaIdx::ion_lev][pidx];

        }
    });
}
    }

    AMREX_ASSERT(OK());
}

void
PlasmaParticleContainer::
InitIonizationModule (const amrex::Geometry& geom,
                      PlasmaParticleContainer* product_pc)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InitIonizationModule()");

    using namespace amrex::literals;

    if (!m_can_ionize) return;
    m_product_pc = product_pc;
    amrex::ParmParse pp(m_name);
    std::string physical_element;
    getWithParser(pp, "element", physical_element);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ion_map_ids.count(physical_element) != 0,
        "There are no ionization energies available for this element. "
        "Please update src/utils/IonizationEnergiesTable.H using write_atomic_data_cpp.py");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE((std::abs(product_pc->m_charge / m_charge +1) < 1e-3),
        "Ion and Ionization product charges have to be opposite");
    // Get atomic number and ionization energies from file
    const int ion_element_id = ion_map_ids[physical_element];
    const int ion_atomic_number = ion_atomic_numbers[ion_element_id];
    amrex::Vector<amrex::Real> h_ionization_energies(ion_atomic_number);
    const int offset = ion_energy_offsets[ion_element_id];
    for(int i=0; i<ion_atomic_number; i++){
        h_ionization_energies[i] = table_ionization_energies[i+offset];
    }
    // Compute ADK prefactors (See Chen, JCP 236 (2013), equation (2))
    // For now, we assume l=0 and m=0.
    // The approximate expressions are used,
    // without Gamma function
    const PhysConst phys_const = make_constants_SI();
    const amrex::Real alpha = 0.0072973525693_rt;
    const amrex::Real r_e = 2.8179403227e-15_rt;
    const amrex::Real a3 = alpha * alpha * alpha;
    const amrex::Real a4 = a3 * alpha;
    const amrex::Real wa = a3 * phys_const.c / r_e;
    const amrex::Real Ea = phys_const.m_e * phys_const.c * phys_const.c / phys_const.q_e * a4 / r_e;
    const amrex::Real UH = table_ionization_energies[0];
    const amrex::Real l_eff = std::sqrt(UH/h_ionization_energies[0]) - 1._rt;
    // partial dx calculation for QSA
    auto dx = geom.CellSizeArray();
    const amrex::Real dt = dx[2] / phys_const.c;

    m_adk_power.resize(ion_atomic_number);
    m_adk_prefactor.resize(ion_atomic_number);
    m_adk_exp_prefactor.resize(ion_atomic_number);

    amrex::Real* AMREX_RESTRICT ionization_energies = h_ionization_energies.data();
    amrex::Real* AMREX_RESTRICT p_adk_power = m_adk_power.data();
    amrex::Real* AMREX_RESTRICT p_adk_prefactor = m_adk_prefactor.data();
    amrex::Real* AMREX_RESTRICT p_adk_exp_prefactor = m_adk_exp_prefactor.data();

    for (int i=0; i<ion_atomic_number; ++i)
    {
        const amrex::Real n_eff = (i+1) * std::sqrt(UH/ionization_energies[i]);
        const amrex::Real C2 = std::pow(2,2*n_eff)/(n_eff*std::tgamma(n_eff+l_eff+1)
                         * std::tgamma(n_eff-l_eff));
        p_adk_power[i] = -(2*n_eff - 1);
        const amrex::Real Uion = ionization_energies[i];
        p_adk_prefactor[i] = dt * wa * C2 * ( Uion/(2*UH) )
            * std::pow(2*std::pow((Uion/UH),3./2)*Ea,2*n_eff - 1);
        p_adk_exp_prefactor[i] = -2./3 * std::pow( Uion/UH,3./2) * Ea;
    }
}
