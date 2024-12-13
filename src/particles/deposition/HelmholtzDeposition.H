/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HELMHOLTZDEPOSITION_H_
#define HELMHOLTZDEPOSITION_H_

#include "particles/beam/BeamParticleContainer.H"
#include "helmholtz/Helmholtz.H"
#include "utils/Constants.H"

void
HelmholtzDeposition (BeamParticleContainer& beam, Helmholtz& helmholtz,
                    const bool do_dtau,
                    const int which_beam_slice,
                    const int islice,
                    const int isubslice);

#endif //  HELMHOLTZDEPOSITION_H_
