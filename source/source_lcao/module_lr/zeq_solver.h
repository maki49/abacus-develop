#pragma once
#include "hamilt_zeq_l.h"
#include "hamilt_zeq_r.h"

// `Z_vector_equation` is defined in zeq_solver.hpp. A declaration used to sit here with an older
// signature (no pot_hxc_gs / openshell / zvec_solver) and no definition anywhere, so any call that
// happened to match it would have failed at link time.

#include "zeq_solver.hpp"