#include "timing.hpp"
#include <chrono>

double now_seconds() {
	using clock = std::chrono::steady_clock;
	const auto since_start = clock::now().time_since_epoch();

	return std::chrono::duration<double>(since_start).count();
}
