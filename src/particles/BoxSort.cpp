#include "BoxSort.H"

#include <AMReX_ParticleTransformation.H>

void BoxSorter::sortParticlesByBox (BeamParticleContainer& a_beam,
                                    const amrex::BoxArray a_ba, const amrex::Geometry& a_geom)
{
    if (! m_particle_locator.isValid(a_ba)) m_particle_locator.build(a_ba, a_geom);
    auto assign_grid = m_particle_locator.getGridAssignor();

    int const np = a_beam.numParticles();
    BeamParticleContainer::ParticleType* particle_ptr = a_beam.GetArrayOfStructs()().data();

    constexpr unsigned int max_unsigned_int = std::numeric_limits<unsigned int>::max();

    int num_boxes = a_ba.size();
    m_box_counts.resize(0);
    m_box_offsets.resize(0);
    m_box_counts.resize(num_boxes+1, 0);
    m_box_offsets.resize(num_boxes+1);

    amrex::Gpu::DeviceVector<unsigned int> dst_indices(np);

    auto p_box_counts = m_box_counts.dataPtr();
    auto p_dst_indices = dst_indices.dataPtr();
    AMREX_FOR_1D ( np, i,
    {
        int dst_box = assign_grid(particle_ptr[i]);
        if (dst_box < 0) {
            // particle has left domain transversely, stick it at the end and invalidate
            dst_box = num_boxes;
            particle_ptr[i].id() = -std::abs(particle_ptr[i].id());
        }
        unsigned int index = amrex::Gpu::Atomic::Inc(
            &p_box_counts[dst_box], max_unsigned_int);
        p_dst_indices[i] = index;
    });

    amrex::Gpu::exclusive_scan(m_box_counts.begin(), m_box_counts.end(), m_box_offsets.begin());

    BeamParticleContainer tmp(a_beam.get_name());
    tmp.resize(np);

    auto p_box_offsets = m_box_offsets.dataPtr();
    AMREX_FOR_1D ( np, i,
    {
        int dst_box = assign_grid(particle_ptr[i]);
        if (dst_box < 0) dst_box = num_boxes;
        p_dst_indices[i] += p_box_offsets[dst_box];
    });

    amrex::scatterParticles(tmp, a_beam, np, dst_indices.dataPtr());

    a_beam.swap(tmp);
}
