// Minimales Test-Gerüst: Registrierung per TEST(name), Prüfungen per CHECK*.
// Kein externes Framework, damit der Build keine weitere Abhängigkeit bekommt.
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct TestCase {
	std::string name;
	std::function<void()> fn;
};

std::vector<TestCase>& AllTests();

struct TestRegistrar {
	TestRegistrar(const char* name, std::function<void()> fn) { AllTests().push_back({ name, fn }); }
};

#define TEST(name)                                          \
	static void name();                                     \
	static TestRegistrar reg_##name(#name, name);           \
	static void name()

#define FAIL_AT(msg)                                                              \
	do {                                                                          \
		std::ostringstream _s;                                                    \
		_s << __FILE__ << ":" << __LINE__ << ": " << msg;                         \
		throw std::runtime_error(_s.str());                                       \
	} while (0)

#define CHECK(cond)                                                               \
	do { if (!(cond)) FAIL_AT("CHECK(" #cond ") fehlgeschlagen"); } while (0)

#define CHECK_NEAR(actual, expected, eps)                                         \
	do {                                                                          \
		double _a = (double)(actual), _e = (double)(expected);                    \
		if (!(std::abs(_a - _e) <= (eps)))                                        \
			FAIL_AT("erwartet " << _e << " +/- " << (eps) << ", war " << _a       \
				<< "  [" #actual "]");                                            \
	} while (0)

#define CHECK_EQ(actual, expected)                                                \
	do {                                                                          \
		auto _a = (actual); auto _e = (expected);                                 \
		if (!(_a == _e))                                                          \
			FAIL_AT("erwartet " << _e << ", war " << _a << "  [" #actual "]");    \
	} while (0)

#define CHECK_GT(a, b)                                                            \
	do {                                                                          \
		double _a = (double)(a), _b = (double)(b);                                \
		if (!(_a > _b)) FAIL_AT(_a << " ist nicht > " << _b << "  [" #a " > " #b "]"); \
	} while (0)
