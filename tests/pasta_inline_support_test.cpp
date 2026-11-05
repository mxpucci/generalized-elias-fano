#include <gtest/gtest.h>
#include <filesystem>
#include "datastructures/PastaBitVector.hpp"

template<class BV>
void exercise_inline_ownership() {
    std::vector<bool> bits(6500);
    std::vector<size_t> ones,zeros;
    for(size_t i=0;i<bits.size();++i) {
        bits[i]=(i%7==0)||(i>4000 && i%3==0);
        (bits[i]?ones:zeros).push_back(i);
    }
    const auto check=[&](const BV& bv) {
        size_t count=0;
        for(size_t i=0;i<bits.size();++i) {
            ASSERT_EQ(bv.rank(i),count);
            ASSERT_EQ(bv[i],bits[i]);
            count+=bits[i];
        }
        for(size_t i=0;i<ones.size();++i) ASSERT_EQ(bv.select(i+1),ones[i]);
        for(size_t i=0;i<zeros.size();++i) ASSERT_EQ(bv.select0(i+1),zeros[i]);
    };
    BV original(bits);original.enable_rank();original.enable_select1();original.enable_select0();
    check(original);
    BV copied(original);check(copied);
    BV moved(std::move(copied));check(moved);
    BV assigned(1);assigned=original;check(assigned);
    BV move_assigned(1);move_assigned=std::move(assigned);check(move_assigned);
    auto cloned=original.clone();
    for(size_t i=0;i<zeros.size();++i) ASSERT_EQ(cloned->select0(i+1),zeros[i]);
    const auto path=std::filesystem::temp_directory_path()/"gef-pasta-inline-support.bin";
    original.serialize(path);auto loaded=BV::load(path);check(loaded);
    std::filesystem::remove(path);
    BV no_support(bits);
    no_support.serialize(path);auto loaded_without=BV::load(path);std::filesystem::remove(path);
    EXPECT_EQ(loaded_without.support_size_in_bytes(),0);
    loaded_without.enable_rank();loaded_without.enable_select1();loaded_without.enable_select0();check(loaded_without);
    const size_t byte_words=(bits.size()/64+1)*8;
    EXPECT_EQ(no_support.allocated_size_in_bytes(),sizeof(BV)+byte_words);
    EXPECT_GE(original.allocated_size_in_bytes(),sizeof(BV)+byte_words);
}

TEST(PastaInlineSupport, CopyMoveAssignmentCloneSerializeAndAccounting) {
    exercise_inline_ownership<PastaBitVector>();
}
