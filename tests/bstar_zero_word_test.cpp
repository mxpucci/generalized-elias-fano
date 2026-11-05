#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "gef/B_STAR_GEF.hpp"

// Test-only MSB-first backend: encoding uses the normal FastBitWriter<true>,
// while independent logical scanning supplies select for range starts/queries.
class ReverseGap : public SDSLBitVector {
public:
 using SDSLBitVector::SDSLBitVector;
 explicit ReverseGap(SDSLBitVector&& other):SDSLBitVector(std::move(other)){}
 static constexpr bool reverse_bit_order=true;
 void enable_select0()override{}
 size_t select0(size_t rank)const override {
  for(size_t i=0;i<size();++i)if(((raw_data_ptr()[i/64]>>(63-(i&63)))&1)==0 && --rank==0)return i;
  throw std::out_of_range("select0");
 }
 size_t select0_unchecked(size_t rank)const{return select0(rank);}
};

template<class Bits>void check_zero_words(){
 for(size_t n:{size_t(1),size_t(63),size_t(64),size_t(65),size_t(127),size_t(128),size_t(255),size_t(256),size_t(257),size_t(511),size_t(512),size_t(513),size_t(1031)}){
  for(unsigned pattern=0;pattern<2;++pattern){
   std::vector<uint64_t> values(n,0);if(pattern)for(size_t i=0;i<n;++i)values[i]=((i/129)%5)*256+(i%16);
   const gef::BlockParameters<uint64_t> parameters{0,pattern?uint8_t(11):uint8_t(1),pattern?uint8_t(4):uint8_t(0)};
   gef::internal::B_STAR_GEF<uint64_t,Bits> codec(values,parameters);
   for(size_t i=0;i<n;++i)ASSERT_EQ(codec[i],values[i]);
   for(size_t start:{size_t(0),size_t(1),size_t(7),size_t(63),size_t(64),size_t(65),size_t(127),size_t(129),size_t(255),size_t(256),size_t(257),size_t(1023)})if(start<n)
    for(size_t requested:{size_t(8),size_t(63),size_t(64),size_t(65),size_t(127),size_t(256),size_t(511)}){
     const size_t count=std::min(requested,n-start);std::vector<uint64_t> out(count+3,UINT64_MAX);
     ASSERT_EQ(codec.get_elements(start,count,out),count);
     ASSERT_TRUE(std::equal(values.begin()+start,values.begin()+start+count,out.begin()));
     ASSERT_EQ(out[count],UINT64_MAX);
    }
  }
 }
}
TEST(BStarZeroWord, LsbFirstCountsStartsAndTails){check_zero_words<PastaGapBitVector>();}
TEST(BStarZeroWord, MsbFirstCountsStartsAndTails){check_zero_words<ReverseGap>();}
