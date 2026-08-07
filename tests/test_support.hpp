#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace kripcuda::testing {

inline int failureCount = 0;

inline void expect(bool condition, const std::string& what, const char* file, int line) {
    if (!condition) {
        ++failureCount;
        std::cerr << file << ':' << line << ": FAILED " << what << '\n';
    }
}

template <typename Lhs, typename Rhs>
void expectEqual(const Lhs& lhs, const Rhs& rhs, const std::string& what, const char* file,
                 int line) {
    if (!(lhs == rhs)) {
        ++failureCount;
        std::cerr << file << ':' << line << ": FAILED " << what << " (" << lhs << " != " << rhs
                  << ")\n";
    }
}

inline int report(const char* suite) {
    if (failureCount == 0) {
        std::cout << suite << ": all checks passed\n";
        return EXIT_SUCCESS;
    }
    std::cerr << suite << ": " << failureCount << " check(s) failed\n";
    return EXIT_FAILURE;
}

} // namespace kripcuda::testing

#define EXPECT_TRUE(condition)                                                                     \
    ::kripcuda::testing::expect((condition), #condition, __FILE__, __LINE__)

#define EXPECT_EQ(lhs, rhs)                                                                        \
    ::kripcuda::testing::expectEqual((lhs), (rhs), #lhs " == " #rhs, __FILE__, __LINE__)

#define EXPECT_THROWS(expression, exception)                                                        \
    do {                                                                                           \
        bool thrown = false;                                                                       \
        try {                                                                                      \
            expression;                                                                            \
        } catch (const exception&) {                                                               \
            thrown = true;                                                                         \
        }                                                                                          \
        ::kripcuda::testing::expect(thrown, #expression " throws " #exception, __FILE__,           \
                                    __LINE__);                                                     \
    } while (false)
