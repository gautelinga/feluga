// From fenicstools (https://github.com/mikaem/fenicstools), Copyright (C) 2011-2013
// Mikael Mortensen; modified in femtools (NOTICE).
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef __PROBES_H
#define __PROBES_H

#include "Probe.h"
#include "ProbeCollection.h"

namespace dolfin
{

  // A Function's values at many points (`ProbeCollection`)
  class Probes : public ProbeCollection<Probe>
  {

  public:

    using ProbeCollection<Probe>::ProbeCollection;

  };

}

#endif
