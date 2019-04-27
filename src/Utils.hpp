#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <typeinfo>

template<class _Ty> inline void writeToBinStream(std::ostream& os, const _Ty& v);
template<class _Ty> inline _Ty readFromBinStream(std::istream& is);
template<class _Ty> inline void readFromBinStream(std::istream& is, _Ty& v);

template<class _Ty>
inline typename std::enable_if<!std::is_fundamental<_Ty>::value && !std::is_enum<_Ty>::value>::type writeToBinStreamImpl(std::ostream& os, const _Ty& v)
{
	static_assert(true, "Only fundamental type can be written!");
}

template<class _Ty>
inline typename std::enable_if<!std::is_fundamental<_Ty>::value && !std::is_enum<_Ty>::value>::type readFromBinStreamImpl(std::ostream& os, const _Ty& v)
{
	static_assert(true, "Only fundamental type can be read!");
}

template<class _Ty>
inline typename std::enable_if<std::is_fundamental<_Ty>::value || std::is_enum<_Ty>::value>::type writeToBinStreamImpl(std::ostream& os, const _Ty& v)
{
	if (!os.write((const char*)&v, sizeof(_Ty))) throw std::ios_base::failure(std::string{ "writing type '" } + typeid(_Ty).name() + "' failed");
}

template<class _Ty>
inline typename std::enable_if<std::is_fundamental<_Ty>::value || std::is_enum<_Ty>::value>::type readFromBinStreamImpl(std::istream& is, _Ty& v)
{
	if (!is.read((char*)&v, sizeof(_Ty))) throw std::ios_base::failure(std::string{ "reading type '" } +typeid(_Ty).name() + "' failed");
}

template<class _Ty1, class _Ty2>
inline void writeToBinStreamImpl(std::ostream& os, const typename std::pair<_Ty1, _Ty2>& v)
{
	writeToBinStream(os, v.first);
	writeToBinStream(os, v.second);
}

template<class _Ty1, class _Ty2>
inline void readFromBinStreamImpl(std::istream& is, typename std::pair<_Ty1, _Ty2>& v)
{
	v.first = readFromBinStream<_Ty1>(is);
	v.second = readFromBinStream<_Ty2>(is);
}


template<class _Ty1, class _Ty2>
inline void writeToBinStreamImpl(std::ostream& os, const typename std::map<_Ty1, _Ty2>& v)
{
	writeToBinStream<uint32_t>(os, v.size());
	for (auto& p : v)
	{
		writeToBinStream(os, p);
	}
}

template<class _Ty1, class _Ty2>
inline void readFromBinStreamImpl(std::istream& is, typename std::map<_Ty1, _Ty2>& v)
{
	size_t len = readFromBinStream<uint32_t>(is);
	v.clear();
	for (size_t i = 0; i < len; ++i)
	{
		v.emplace(readFromBinStream<std::pair<_Ty1, _Ty2>>(is));
	}
}

template<class _Ty>
inline void writeToBinStream(std::ostream& os, const _Ty& v)
{
	writeToBinStreamImpl(os, v);
}


template<class _Ty>
inline _Ty readFromBinStream(std::istream& is)
{
	_Ty v;
	readFromBinStreamImpl(is, v);
	return v;
}

template<class _Ty>
inline void readFromBinStream(std::istream& is, _Ty& v)
{
	readFromBinStreamImpl(is, v);
}

inline uint32_t readVFromBinStream(std::istream & is)
{
	static uint32_t vSize[] = { 0, 0x80, 0x4080, 0x204080, 0x10204080 };
	char c;
	uint32_t v = 0;
	size_t i;
	for (i = 0; (c = readFromBinStream<uint8_t>(is)) & 0x80; ++i)
	{
		v |= (c & 0x7F) << (i * 7);
	}
	v |= c << (i * 7);
	return v + vSize[i];
}

inline void writeVToBinStream(std::ostream & os, uint32_t v)
{
	static uint32_t vSize[] = { 0, 0x80, 0x4080, 0x204080, 0x10204080 };
	size_t i;
	for (i = 1; i <= 4; ++i)
	{
		if (v < vSize[i]) break;
	}
	v -= vSize[i - 1];
	for (size_t n = 0; n < i; ++n)
	{
		uint8_t c = (v & 0x7F) | (n + 1 < i ? 0x80 : 0);
		writeToBinStream(os, c);
		v >>= 7;
	}
}

inline int32_t readSVFromBinStream(std::istream & is)
{
	static int32_t vSize[] = { 0x40, 0x2000, 0x100000, 0x8000000 };
	char c;
	uint32_t v = 0;
	size_t i;
	for (i = 0; (c = readFromBinStream<uint8_t>(is)) & 0x80; ++i)
	{
		v |= (c & 0x7F) << (i * 7);
	}
	v |= c << (i * 7);
	if (i >= 4) return (int32_t)v;
	return v - (v >= vSize[i] ? (1 << ((i + 1) * 7)) : 0);
}

inline void writeSVToBinStream(std::ostream & os, int32_t v)
{
	static int32_t vSize[] = { 0, 0x40, 0x2000, 0x100000, 0x8000000 };
	size_t i;
	for (i = 1; i <= 4; ++i)
	{
		if (-vSize[i] <= v && v < vSize[i]) break;
	}
	uint32_t u;
	if (i >= 5) u = (uint32_t)v;
	else u = v + (v < 0 ? (1 << (i * 7)) : 0);
	for (size_t n = 0; n < i; ++n)
	{
		uint8_t c = (u & 0x7F) | (n + 1 < i ? 0x80 : 0);
		writeToBinStream(os, c);
		u >>= 7;
	}
}
