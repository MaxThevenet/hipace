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
HelmholtzDepositon (BeamParticleContainer& beam, Helmholtz& helmholtz,
                    const bool do_dtau,
                    const int which_beam_slice);

#endif //  HELMHOLTZDEPOSITION_H_
