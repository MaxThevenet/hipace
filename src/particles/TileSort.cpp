/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#include "TileSort.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_ParticleTransformation.H>

PlasmaBins
findParticlesInEachTile (
    int lev, amrex::Box bx, int bin_size, int tile_size,
    PlasmaParticleContainer& plasma, const amrex::Geometry& geom)
{
    HIPACE_PROFILE("findParticlesInEachTile()");

    // Tile box: only 1 cell longitudinally, same as bx transversally.
    const int s = tile_size / bin_size;
    const amrex::Box tcbx = bx.coarsen(bin_size);
    const amrex::Box cbx = {{tcbx.smallEnd(0),tcbx.smallEnd(1),0}, {tcbx.bigEnd(0),tcbx.bigEnd(1),0}};

    PlasmaBins bins;

    // Extract particle structures for this tile
    int count = 0; // number of boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti) {
        count += 1;

        // Extract box properties
        const auto lo = lbound(cbx);
        const auto hi = ubound(cbx);
        const auto dxi = amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>({
                geom.InvCellSizeArray()[0]/bin_size,
                geom.InvCellSizeArray()[1]/bin_size,
                1.});
        const auto plo = geom.ProbLoArray();
        const int nx = hi.x-lo.x+1;
        const int ny = hi.y-lo.y+1;
        
        // Find the particles that are in each tile and return collections of indices per tile.
        bins.build(
            pti.numParticles(), pti.GetArrayOfStructs().begin(), nx*ny,
            // Pass lambda function that returns the tile index
            [=] AMREX_GPU_DEVICE (const PlasmaParticleContainer::ParticleType& p)
            noexcept -> int
            {
                const int ix = static_cast<int>((p.pos(0)-plo[0])*dxi[0]-lo.x);
                const int iy = static_cast<int>((p.pos(1)-plo[1])*dxi[1]-lo.y);
                return (ix/s) * s * ny + (iy/s) * s * s + (iy%s) + (ix%s)*s;
            });
    }
    AMREX_ALWAYS_ASSERT(count <= 1);
    return bins;
}
