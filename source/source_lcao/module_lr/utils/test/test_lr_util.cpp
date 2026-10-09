#include "../lr_util.h"
#include "source_cell/unitcell.h"
#include <gtest/gtest.h>
#include <vector>

TEST(LRForcePP, AllowsForcesAndRelaxationWithoutNLCC)
{
    UnitCell cell;
    cell.ntype = 1;
    cell.atoms = new Atom[1];
    cell.set_atom_flag = true;
    EXPECT_NO_THROW(LR_Util::check_force_pp(cell, true, "scf"));
    EXPECT_NO_THROW(LR_Util::check_force_pp(cell, true, "relax"));
}

TEST(LRForcePP, AllowsNLCCTDDFTSpectra)
{
    UnitCell cell;
    cell.ntype = 2;
    cell.atoms = new Atom[2];
    cell.set_atom_flag = true;
    cell.pseudo_fn = {"H.upf", "Si_NLCC.upf"};
    cell.atoms[1].ncpp.nlcc = true;
    EXPECT_NO_THROW(LR_Util::check_force_pp(cell, false, "scf"));
    EXPECT_NO_THROW(LR_Util::check_force_pp(cell, false, "nscf"));
}

TEST(LRForcePP, RejectsNLCCForcesAndIdentifiesThePseudopotential)
{
    UnitCell cell;
    cell.ntype = 2;
    cell.atoms = new Atom[2];
    cell.set_atom_flag = true;
    cell.pseudo_fn = {"H.upf", "Si_NLCC.upf"};
    cell.atoms[1].ncpp.nlcc = true;
    cell.atoms[1].label = "Si";
    try
    {
        LR_Util::check_force_pp(cell, true, "scf");
        FAIL() << "NLCC forces were accepted";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("element Si"), std::string::npos);
        EXPECT_NE(message.find("Si_NLCC.upf"), std::string::npos);
        EXPECT_NE(message.find("not implemented"), std::string::npos);
    }
}

TEST(LRForcePP, RejectsNLCCRelaxationEvenWithoutExplicitForceFlag)
{
    UnitCell cell;
    cell.ntype = 2;
    cell.atoms = new Atom[2];
    cell.set_atom_flag = true;
    cell.pseudo_fn = {"H.upf", "Si_NLCC.upf"};
    cell.atoms[1].ncpp.nlcc = true;
    EXPECT_THROW(LR_Util::check_force_pp(cell, false, "relax"), std::invalid_argument);
    EXPECT_THROW(LR_Util::check_force_pp(cell, true, "relax"), std::invalid_argument);
}
