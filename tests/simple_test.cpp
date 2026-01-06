#include <gtest/gtest.h>
#include "gef/gef.hpp"

TEST(GefTest, Hello) {
    // Smoke test: basic construction works.
    const std::vector<int64_t> data = {1, 2, 3, 10, 20, 21};
    gef::internal::B_GEF<int64_t> bgef(data);
    EXPECT_EQ(bgef.size(), data.size());
    EXPECT_EQ(bgef[0], data[0]);
} 