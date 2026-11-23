#pragma once
struct __half { unsigned short x; };
__attribute__((device)) float __half2float(__half h);
__attribute__((device)) __half __float2half(float f);
__attribute__((device)) __half __ushort_as_half(unsigned short v);
__attribute__((device)) unsigned short __half_as_ushort(__half h);
