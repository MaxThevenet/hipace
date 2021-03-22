#include "BeamDepositCurrent.H"
#include "particles/BeamParticleContainer.H"
#include "particles/deposition/BeamDepositCurrentInner.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_DenseBins.H>

void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields, amrex::Geometry const& gm,
                     int const lev ,const int islice, const amrex::Box bx, int const offset,
                     amrex::DenseBins<BeamParticleContainer::ParticleType>& bins,
                     const bool do_beam_jx_jy_deposition)
{
    HIPACE_PROFILE("DepositCurrentSlice_BeamParticleContainer()");
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Hipace::m_depos_order_z == 0,
        "Only order 0 deposition is allowed for beam per-slice deposition");

    PhysConst const phys_const = get_phys_const();

    // Assumes '2' == 'z' == 'the long dimension'.
    int islice_local = islice - bx.smallEnd(2);

    // Extract properties associated with the extent of the current box
    amrex::Box tilebox = bx;
    tilebox.grow({Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, Hipace::m_depos_order_z});

    amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
    amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
    amrex::Dim3 const lo = amrex::lbound(tilebox);

    // Extract the fields currents
    amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
    amrex::MultiFab jx(S, amrex::make_alias, Comps[WhichSlice::This]["jx"], 1);
    amrex::MultiFab jy(S, amrex::make_alias, Comps[WhichSlice::This]["jy"], 1);
    amrex::MultiFab jz(S, amrex::make_alias, Comps[WhichSlice::This]["jz"], 1);

    // Extract FabArray for this box (because there is currently no transverse
    // parallelization, the index we want in the slice multifab is always 0.
    // Fix later.
    amrex::FArrayBox& jx_fab = jx[0];
    amrex::FArrayBox& jy_fab = jy[0];
    amrex::FArrayBox& jz_fab = jz[0];

    // For now: fix the value of the charge
    const amrex::Real q = - phys_const.q_e;

    // Call deposition function in each box
    if        (Hipace::m_depos_order_xy == 0){
        doDepositionShapeN<0, 0>( beam, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, islice_local,
                                  bins, offset, do_beam_jx_jy_deposition);
    } else if (Hipace::m_depos_order_xy == 1){
        doDepositionShapeN<1, 0>( beam, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, islice_local,
                                  bins, offset, do_beam_jx_jy_deposition);
    } else if (Hipace::m_depos_order_xy == 2){
        doDepositionShapeN<2, 0>( beam, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, islice_local,
                                  bins, offset, do_beam_jx_jy_deposition);
    } else if (Hipace::m_depos_order_xy == 3){
        doDepositionShapeN<3, 0>( beam, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, islice_local,
                                  bins, offset, do_beam_jx_jy_deposition);
    } else {
        amrex::Abort("unknown deposition order");
    }
}
