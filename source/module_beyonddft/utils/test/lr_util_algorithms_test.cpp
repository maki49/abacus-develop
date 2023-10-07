#include <gtest/gtest.h>

#include "../lr_util_algorithms.hpp"

TEST(LR_Util, PsiWrapper)
{
    int nk = 2;
    int nbands = 5;
    int nbasis = 6;

    psi::Psi<float> k1(1, nbands, nk * nbasis);
    for (int i = 0;i < nbands * nk * nbasis;++i)k1.get_pointer()[i] = i;

    k1.fix_b(2);
    psi::Psi<float> bf = LR_Util::k1_to_bfirst_wrapper(k1, nk, nbasis);
    EXPECT_EQ(k1.get_current_k(), 0);
    EXPECT_EQ(k1.get_current_b(), 2); // invariance after wrapper
    EXPECT_EQ(bf.get_current_k(), 0);
    EXPECT_EQ(bf.get_current_b(), 0);

    bf.fix_kb(1, 3);
    psi::Psi<float> kb = LR_Util::bfirst_to_k1_wrapper(bf);
    EXPECT_EQ(bf.get_current_k(), 1);
    EXPECT_EQ(bf.get_current_b(), 3);
    EXPECT_EQ(kb.get_current_k(), 0);
    EXPECT_EQ(kb.get_current_b(), 0);


    EXPECT_EQ(bf.get_k_first(), false);
    EXPECT_EQ(bf.get_nk(), nk);
    EXPECT_EQ(bf.get_nbands(), nbands);
    EXPECT_EQ(bf.get_nbasis(), nbasis);

    EXPECT_EQ(kb.get_k_first(), true);
    EXPECT_EQ(kb.get_nk(), 1);
    EXPECT_EQ(kb.get_nbands(), nbands);
    EXPECT_EQ(kb.get_nbasis(), nk * nbasis);

    k1.fix_b(0);
    bf.fix_kb(0, 0);
    EXPECT_EQ(bf.get_pointer(), k1.get_pointer());
    EXPECT_EQ(bf.get_pointer(), kb.get_pointer());
    for (int ik = 0; ik < nk; ik++)
    {
        for (int ib = 0; ib < nbands; ib++)
        {
            bf.fix_kb(ik, ib);
            kb.fix_b(ib);
            k1.fix_b(ib);
            for (int ibasis = 0; ibasis < nbasis; ibasis++)
            {
                int ikb = ik * nbasis + ibasis;
                EXPECT_EQ(kb(ikb), bf(ibasis));
                EXPECT_EQ(k1(ikb), kb(ikb));
            }
        }
    }
}
