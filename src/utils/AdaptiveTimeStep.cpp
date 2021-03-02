#include "AdaptiveTimeStep.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Constants.H"

#ifdef AMREX_USE_MPI
namespace {
    constexpr int comm_z_tag = 2000;
}
#endif

/** \brief describes which double is used for the adaptive time step */
struct WhichDouble {
    enum Comp { Dt=0, MinUz, SumWeights, SumWeightsTimesUz, SumWeightsTimesUzSquared, N };
};

AdaptiveTimeStep::AdaptiveTimeStep ()
{
    amrex::ParmParse ppa("hipace");
    ppa.query("do_adaptive_time_step", m_do_adaptive_time_step);
    ppa.query("nt_per_omega_betatron", m_nt_per_omega_betatron);
}

void
AdaptiveTimeStep::Calculate (amrex::Real& dt, MultiBeam& beams, PlasmaParticleContainer& plasma,
                             const int it, const amrex::Vector<BoxSorter>& a_box_sorter_vec)
{
    HIPACE_PROFILE("CalculateAdaptiveTimeStep()");

    if (m_do_adaptive_time_step == 0) return;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( plasma.m_density != 0,
        "A plasma density must be specified to use an adaptive time step.");

    // Extract properties associated with physical size of the box
    const PhysConst phys_const = get_phys_const();

    for (int ibeam = 0; ibeam < beams.get_nbeams(); ibeam++) {
        const auto& beam = beams.getBeam(ibeam);

        const uint64_t box_offset = a_box_sorter_vec[ibeam].boxOffsetsPtr()[it];
        const uint64_t numParticleOnTile = a_box_sorter_vec[ibeam].boxCountsPtr()[it];


        // Extract particle properties
        const auto& soa = beam.GetStructOfArrays(); // For momenta and weights
        const auto uzp = soa.GetRealData(BeamIdx::uz).data() + box_offset;
        const auto wp = soa.GetRealData(BeamIdx::w).data() + box_offset;

        amrex::Gpu::DeviceScalar<amrex::Real> gpu_min_uz(m_timestep_data[WhichDouble::MinUz]);
        amrex::Real* p_min_uz = gpu_min_uz.dataPtr();

        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights(
                m_timestep_data[WhichDouble::SumWeights]);
        amrex::Real* p_sum_weights = gpu_sum_weights.dataPtr();

        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights_times_uz(
            m_timestep_data[WhichDouble::SumWeightsTimesUz]);
        amrex::Real* p_sum_weights_times_uz = gpu_sum_weights_times_uz.dataPtr();

        amrex::Gpu::DeviceScalar<amrex::Real> gpu_sum_weights_times_uz_squared(
            m_timestep_data[WhichDouble::SumWeightsTimesUzSquared]);
        amrex::Real* p_sum_weights_times_uz_squared =
            gpu_sum_weights_times_uz_squared.dataPtr();

        amrex::ParallelFor(numParticleOnTile,
            [=] AMREX_GPU_DEVICE (long ip) {

                if ( std::abs(wp[ip]) < std::numeric_limits<amrex::Real>::epsilon() ) return;

                amrex::Gpu::Atomic::Add(p_sum_weights, wp[ip]);
                amrex::Gpu::Atomic::Add(p_sum_weights_times_uz, wp[ip]*uzp[ip]/phys_const.c);
                amrex::Gpu::Atomic::Add(p_sum_weights_times_uz_squared, wp[ip]*uzp[ip]*uzp[ip]
                                        /phys_const.c/phys_const.c);
                amrex::Gpu::Atomic::Min(p_min_uz, uzp[ip]/phys_const.c);

        }
        );
        /* adding beam particle information to time step info */
        m_timestep_data[WhichDouble::SumWeights] = gpu_sum_weights.dataValue();
        m_timestep_data[WhichDouble::SumWeightsTimesUz] = gpu_sum_weights_times_uz.dataValue();
        m_timestep_data[WhichDouble::SumWeightsTimesUzSquared] =
                                               gpu_sum_weights_times_uz_squared.dataValue();
        m_timestep_data[WhichDouble::MinUz] = std::min(m_timestep_data[WhichDouble::MinUz],
                                               gpu_min_uz.dataValue());
    }

    /* only the last box calculates the adaptive time step from the full beam information */
    if (it == 0 )
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_timestep_data[WhichDouble::SumWeights] != 0,
            "The sum of all weights is 0! Probably no beam particles are initialized");
        const amrex::Real mean_uz = m_timestep_data[WhichDouble::SumWeightsTimesUz]
                                       /m_timestep_data[WhichDouble::SumWeights];
        const amrex::Real sigma_uz = sqrt(m_timestep_data[WhichDouble::SumWeightsTimesUzSquared]
                                          /m_timestep_data[WhichDouble::SumWeights] - mean_uz);
        const amrex::Real sigma_uz_dev = mean_uz - 4.*sigma_uz;
        const amrex::Real max_supported_uz = 1e30;
        const amrex::Real chosen_min_uz = std::min( std::max(sigma_uz_dev,
                                          m_timestep_data[WhichDouble::MinUz]), max_supported_uz);

        if (Hipace::m_verbose >=2 ){
            amrex::Print()<<"Minimum gamma to calculate new time step: " << chosen_min_uz << "\n";
        }

        if (chosen_min_uz < 1) {
            amrex::Print()<<"WARNING: beam particles have non-relativistic velocities!";
        }

        amrex::Real new_dt = dt;
        if (chosen_min_uz > 1) // and density above min density
        {
            const amrex::Real omega_p = sqrt(plasma.m_density * phys_const.q_e*phys_const.q_e
                                          / ( phys_const.ep0*phys_const.m_e ));
            new_dt = sqrt(2.*chosen_min_uz)/omega_p * m_nt_per_omega_betatron;
        }

        /* set the new time step */
        dt = new_dt;

    }
}
