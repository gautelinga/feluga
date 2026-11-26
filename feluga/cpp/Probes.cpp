// From fenicstools (https://github.com/mikaem/fenicstools), Copyright (C) 2011-2013
// Mikael Mortensen; modified in feluga (NOTICE).
// SPDX-License-Identifier: LGPL-3.0-or-later
// Probes is ProbeCollection<Probe>, whose code is in its header; this unit only
// instantiates it once.
#include "Probes.h"

template class dolfin::ProbeCollection<dolfin::Probe>;
