#include <gtest/gtest.h>
#include <fstream>
#include "gef/gef.hpp"
#include "gef/AllocatedSize.hpp"

TEST(AllocatedSize, SdlsPayloadIncludesPaddingAndCapacity) {
    sdsl::int_vector<> vector(64,0,1);
    EXPECT_GE(gef::internal::sdsl_allocated_payload_bytes(vector),16);
    EXPECT_LT(sdsl::size_in_bytes(vector),sizeof(vector)+gef::internal::sdsl_allocated_payload_bytes(vector));
}

TEST(AllocatedSize, UniformDoesNotCountEmbeddedCodecsTwice) {
    constexpr size_t K=32000;
    using Codec=gef::internal::B_STAR_GEF<int64_t>;
    using Partitioned=gef::UniformPartitioning<int64_t,Codec,K>;
    std::vector<int64_t> data(2*K);
    for(size_t i=0;i<data.size();++i)data[i]=i/3+(i%7);
    Partitioned all(data);
    Codec first(gef::Span<const int64_t>(data.data(),K));
    Codec second(gef::Span<const int64_t>(data.data()+K,K));
    const size_t expected=sizeof(Partitioned)+2*sizeof(std::optional<Codec>)+first.allocated_size_in_bytes()+second.allocated_size_in_bytes()-2*sizeof(Codec);
    EXPECT_EQ(all.allocated_size_in_bytes(),expected);
    EXPECT_GT(all.allocated_size_in_bytes(),all.size_in_bytes());
    auto moved=std::move(all);
    EXPECT_EQ(moved.allocated_size_in_bytes(),expected);
}

TEST(AllocatedSize, EmptyAndConstantCodecsRemainSmall) {
    std::vector<int64_t> data(32000,7);
    gef::RLE_GEF<int64_t> rle(data);
    gef::B_STAR_GEF_APPROXIMATE<int64_t> bstar(data);
    EXPECT_LT(rle.allocated_size_in_bytes(),512);
    EXPECT_LT(bstar.allocated_size_in_bytes(),512);
    gef::RLE_GEF<int64_t> empty(std::vector<int64_t>{});
    EXPECT_EQ(empty.allocated_size_in_bytes(),sizeof(empty));
}
