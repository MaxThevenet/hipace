#include "PlasmaParticleContainer.H"
#include "Constants.H"
#include "ParticleUtil.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"

using namespace amrex;

void
PlasmaParticleContainer::
InitParticles (const IntVect& a_num_particles_per_cell,
               const amrex::RealVect& a_u_std,
               const amrex::RealVect& a_u_mean,
               const amrex::Real a_density,
               const amrex::Real a_radius,
               const Geometry& a_geom,
               const RealBox& a_bounds)
{
    HIPACE_PROFILE("PlasmaParticleContainer::InitParticles");

    const int lev = 0;
    const auto dx = a_geom.CellSizeArray();
    const auto plo = a_geom.ProbLoArray();

    const int num_ppc = AMREX_D_TERM( a_num_particles_per_cell[0],
                                      *a_num_particles_per_cell[1],
                                      *a_num_particles_per_cell[2]);
    const Real scale_fac = Hipace::m_normalized_units? 1./num_ppc : dx[0]*dx[1]*dx[2]/num_ppc;

    for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.tilebox();

        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        Gpu::ManagedVector<unsigned int> counts(tile_box.numPts(), 0);
        unsigned int* pcount = counts.dataPtr();

        Gpu::ManagedVector<unsigned int> offsets(tile_box.numPts());
        unsigned int* poffset = offsets.dataPtr();

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                Real r[3];

                ParticleUtil::get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                Real x = plo[0] + (i + r[0])*dx[0];
                Real y = plo[1] + (j + r[1])*dx[1];
                Real z = plo[2] + (k + r[2])*dx[2];

                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    z >= a_bounds.hi(2) || z < a_bounds.lo(2) ||
                    x*x + y*y > a_radius*a_radius ) continue;

                int ix = i - lo.x;
                int iy = j - lo.y;
                int iz = k - lo.z;
                int nx = hi.x-lo.x+1;
                int ny = hi.y-lo.y+1;
                int nz = hi.z-lo.z+1;
                unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
                unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
                unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
                unsigned int cellid = (uix * ny + uiy) * nz + uiz;
                pcount[cellid] += 1;
            }
        });

        Gpu::exclusive_scan(counts.begin(), counts.end(), offsets.begin());

        int num_to_add = offsets[tile_box.numPts()-1] + counts[tile_box.numPts()-1];

        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        auto old_size = particle_tile.GetArrayOfStructs().size();
        auto new_size = old_size + num_to_add;
        particle_tile.resize(new_size);

        if (num_to_add == 0) continue;

        ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();

        auto arrdata = particle_tile.GetStructOfArrays().realarray();

        int procID = ParallelDescriptor::MyProc();
        int pid = ParticleType::NextID();
        ParticleType::NextID(pid + num_to_add);

        PhysConst phys_const = get_phys_const();

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
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
            unsigned int cellid = (uix * ny + uiy) * nz + uiz;

            int pidx = int(poffset[cellid] - poffset[0]);

            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                Real r[3] = {0.,0.,0.};

                ParticleUtil::get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                Real x = plo[0] + (i + r[0])*dx[0];
                Real y = plo[1] + (j + r[1])*dx[1];
                Real z = plo[2] + (k + r[2])*dx[2];

                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    z >= a_bounds.hi(2) || z < a_bounds.lo(2) ||
                    x*x + y*y > a_radius*a_radius ) continue;

                Real u[3] = {0.,0.,0.};
                ParticleUtil::get_gaussian_random_momentum(u, a_u_mean,
                                                           a_u_std);

                ParticleType& p = pstruct[pidx];
                p.id()   = pid + pidx;
                p.cpu()  = procID;
                p.pos(0) = x;
                p.pos(1) = y;
                p.pos(2) = z;

                arrdata[PlasmaIdx::w        ][pidx] = a_density * scale_fac;
                arrdata[PlasmaIdx::ux       ][pidx] = u[0] * phys_const.c;
                arrdata[PlasmaIdx::uy       ][pidx] = u[1] * phys_const.c;
                arrdata[PlasmaIdx::psi      ][pidx] = 0.;
                arrdata[PlasmaIdx::x_prev   ][pidx] = 0.;
                arrdata[PlasmaIdx::y_prev   ][pidx] = 0.;
                arrdata[PlasmaIdx::w_temp   ][pidx] = 0.;
                arrdata[PlasmaIdx::ux_temp  ][pidx] = u[0] * phys_const.c;
                arrdata[PlasmaIdx::uy_temp  ][pidx] = u[1] * phys_const.c;
                arrdata[PlasmaIdx::psi_temp ][pidx] = 0.;
                arrdata[PlasmaIdx::Fx1      ][pidx] = 0.;
                arrdata[PlasmaIdx::Fx2      ][pidx] = 0.;
                arrdata[PlasmaIdx::Fx3      ][pidx] = 0.;
                arrdata[PlasmaIdx::Fy1      ][pidx] = 0.;
                arrdata[PlasmaIdx::Fy2      ][pidx] = 0.;
                arrdata[PlasmaIdx::Fy3      ][pidx] = 0.;
                arrdata[PlasmaIdx::Fux1     ][pidx] = 0.;
                arrdata[PlasmaIdx::Fux2     ][pidx] = 0.;
                arrdata[PlasmaIdx::Fux3     ][pidx] = 0.;
                arrdata[PlasmaIdx::Fuy1     ][pidx] = 0.;
                arrdata[PlasmaIdx::Fuy2     ][pidx] = 0.;
                arrdata[PlasmaIdx::Fuy3     ][pidx] = 0.;
                arrdata[PlasmaIdx::Fpsi1    ][pidx] = 0.;
                arrdata[PlasmaIdx::Fpsi2    ][pidx] = 0.;
                arrdata[PlasmaIdx::Fpsi3    ][pidx] = 0.;
                arrdata[PlasmaIdx::x0       ][pidx] = x;
                arrdata[PlasmaIdx::y0       ][pidx] = y;
                ++pidx;
            }
        });
    }

    AMREX_ASSERT(OK());
}
