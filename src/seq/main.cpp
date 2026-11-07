#include <cstdio>

#include "config.hpp"

int main(int argc, char** argv) {
	const Options o = parse_args(argc, argv);
	std::printf("n=%zu iters=%d reps=%d init=%s dump='%s' header=%d\n", o.n, o.iters, o.reps,
				o.init == InitKind::Top ? "top" : "pattern", o.dump.c_str(), o.header ? 1 : 0);

	return 0;
}
