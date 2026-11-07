#include "config.hpp"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

[[noreturn]] void fail(const std::string& message) {
	std::fprintf(stderr, "erro: %s\n", message.c_str());
	std::exit(2);
}

// Devolve o valor da opção em argv[a], avançando `a`. Uma opção seguinte
// ("--algo") conta como valor ausente, mas um número negativo ("-1") não.
const char* next_value(int argc, char** argv, int& a) {
	const std::string option = argv[a];
	if (a + 1 >= argc || std::string(argv[a + 1]).rfind("--", 0) == 0) {
		fail("falta o valor de " + option);
	}
	return argv[++a];
}

// Converte o texto inteiro em um número dentro de [min, max].
long long parse_integer(const std::string& option, const std::string& text, long long min,
						long long max) {
	long long value = 0;
	std::size_t used = 0;
	try {
		value = std::stoll(text, &used);
	} catch (const std::invalid_argument&) {
		fail(option + " precisa ser um número inteiro: " + text);
	} catch (const std::out_of_range&) {
		fail(option + " está fora do intervalo: " + text);
	}
	if (used != text.size()) {
		fail(option + " precisa ser um número inteiro: " + text);
	}
	if (value < min || value > max) {
		fail(option + " precisa estar entre " + std::to_string(min) + " e " + std::to_string(max));
	}
	return value;
}

} // namespace

Options parse_args(int argc, char** argv) {
	Options o;
	for (int a = 1; a < argc; a++) {
		const std::string arg = argv[a];
		if (arg == "--n") {
			o.n = static_cast<std::size_t>(parse_integer(arg, next_value(argc, argv, a), 3, LLONG_MAX));
		} else if (arg == "--iters") {
			o.iters = static_cast<int>(parse_integer(arg, next_value(argc, argv, a), 0, INT_MAX));
		} else if (arg == "--reps") {
			o.reps = static_cast<int>(parse_integer(arg, next_value(argc, argv, a), 1, INT_MAX));
		} else if (arg == "--init") {
			const std::string kind = next_value(argc, argv, a);
			if (kind == "pattern") {
				o.init = InitKind::Pattern;
			} else if (kind == "top") {
				o.init = InitKind::Top;
			} else {
				fail("--init precisa ser pattern ou top: " + kind);
			}
		} else if (arg == "--dump") {
			o.dump = next_value(argc, argv, a);
		} else if (arg == "--header") {
			o.header = true;
		} else {
			fail("argumento desconhecido: " + arg);
		}
	}
	return o;
}
