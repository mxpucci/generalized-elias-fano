#include <gtest/gtest.h>
#include "gef/gef.hpp"

TEST(GefTest, Hello) {
    // A simple test to check if the library function can be called.
    ASSERT_NO_THROW(gef::hello());
} 