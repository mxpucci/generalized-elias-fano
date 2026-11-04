#include <gtest/gtest.h>
#include <filesystem>
#include "datastructures/PastaBitVector.hpp"
#include "gef/B_STAR_GEF.hpp"
#include "gef/UniformPartitioning.hpp"

template<class T>
concept HasUncheckedSelect0 = requires(const T& bv) { bv.select0_unchecked(size_t(1)); };
template<class T>
concept HasUncheckedSelect1 = requires(const T& bv) { bv.select_unchecked(size_t(1)); };
static_assert(HasUncheckedSelect0<PastaRankBitVector>);
static_assert(HasUncheckedSelect1<PastaRankBitVector>);
static_assert(HasUncheckedSelect0<PastaGapBitVector>);
static_assert(HasUncheckedSelect1<PastaExceptionBitVector>);
template<class T>
concept HasPairedSelect0 = requires(const T& bv) { bv.select0_pair_unchecked(bv,size_t(1)); };
static_assert(!HasPairedSelect0<PastaRankBitVector>);
static_assert(HasPairedSelect0<PastaGapBitVector>);

TEST(PastaRankOnly, CopyMoveAssignmentCloneAndLoad) {
    for(size_t n:{size_t(1),size_t(64),size_t(4095),size_t(4096),size_t(6500),size_t(32000)}) {
        SCOPED_TRACE(n);
        std::vector<bool> bits(n);
        for(size_t i=0;i<n;++i) bits[i]=(i%7==0)||(i>4000 && i%3==0);
        const auto check=[&](const auto& bv) {
            size_t count=0;
            for(size_t i=0;i<=n;++i) {
                ASSERT_EQ(bv.rank(i),count);
                if(i<n){ASSERT_EQ(bv[i],bits[i]);count+=bits[i];}
            }
        };
        PastaRankBitVector original(bits);original.enable_rank();check(original);
        PastaRankBitVector copied(original);check(copied);
        PastaRankBitVector moved(std::move(copied));check(moved);
        PastaRankBitVector assigned(1);assigned=original;check(assigned);
        PastaRankBitVector move_assigned(1);move_assigned=std::move(assigned);check(move_assigned);
        auto cloned=original.clone();check(*cloned);
        const auto path=std::filesystem::temp_directory_path()/"gef-pasta-rank-only.bin";
        original.serialize(path);auto loaded=PastaRankBitVector::load(path);check(loaded);
        std::filesystem::remove(path);
        PastaRankBitVector no_support(bits);
        no_support.serialize(path);auto loaded_without=PastaRankBitVector::load(path);std::filesystem::remove(path);
        EXPECT_EQ(loaded_without.support_size_in_bytes(),0);
        loaded_without.enable_rank();check(loaded_without);
        EXPECT_THROW(original.select(1),std::runtime_error);
        EXPECT_THROW(original.select0(1),std::runtime_error);
        EXPECT_THROW(original.select_unchecked(1),std::runtime_error);
        EXPECT_THROW(original.select0_unchecked(1),std::runtime_error);
        const size_t byte_words=(n/64+1)*8;
        EXPECT_EQ(no_support.allocated_size_in_bytes(),sizeof(PastaRankBitVector)+byte_words);
        using Rank=pasta::FlatRank<pasta::OptimizedFor::DONT_CARE,pasta::BitVector>;
        const size_t directory_bytes=((n/64+1)/64+1)*sizeof(pasta::BigL12Type);
        EXPECT_EQ(original.support_size_in_bytes(),sizeof(Rank)+directory_bytes);
        EXPECT_EQ(original.allocated_size_in_bytes(),sizeof(PastaRankBitVector)+byte_words+directory_bytes);
    }
}

TEST(PastaRankOnly, SupportFreeBStarRetainsGenericBackendCompatibility) {
    using Codec=gef::internal::B_STAR_GEF<int64_t,PastaRankBitVector,false>;
    std::vector<int64_t> values(1025);
    for(size_t i=0;i<values.size();++i)
        values[i]=100000+static_cast<int64_t>(i)*7-static_cast<int64_t>(i%11)*13;
    const auto check=[&](const auto& codec) {
        std::vector<int64_t> output(values.size(),-1);
        EXPECT_EQ(codec.get_elements(0,values.size(),output),values.size());
        EXPECT_EQ(output,values);
        for(size_t i:{size_t(0),size_t(1),size_t(126),size_t(127),size_t(513),size_t(1024)})
            EXPECT_EQ(codec[i],values[i]);
        for(size_t start:{size_t(1),size_t(126),size_t(513),size_t(1024)}) {
            const size_t count=std::min<size_t>(129,values.size()-start);
            std::fill(output.begin(),output.end(),-1);
            EXPECT_EQ(codec.get_elements(start,count,output),count);
            EXPECT_TRUE(std::equal(values.begin()+start,values.begin()+start+count,output.begin()));
            EXPECT_EQ(output[count],-1);
        }
    };
    Codec codec(values);check(codec);
    Codec copied(codec);check(copied);
    Codec moved(std::move(copied));check(moved);
    const auto path=std::filesystem::temp_directory_path()/"gef-bstar-rank-only-no-support.bin";
    codec.serialize(path);Codec loaded;loaded.load(path);std::filesystem::remove(path);check(loaded);
    gef::UniformPartitioning<int64_t,Codec,127> partitioned(values);check(partitioned);
}
