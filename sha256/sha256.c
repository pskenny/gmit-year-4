// Paul Kenny, G00326057, Software Development Year 4, Galway-Mayo Institute Of Technology

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>

// Constants declared in Secion 4.2.2
const uint32_t K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// block of data (512 bits) to process
union block {
	uint64_t sixtyFour[8];
	uint32_t thirtyTwo[16];
	uint8_t eight[64];
};

enum flag
{
	READ,
	PAD,
	PAD1,
	FINISH
};

uint32_t Ch(uint32_t x, uint32_t y, uint32_t z)
{
	// Section 4.1.2
	return (x & y) ^ (~x & z);
}

uint32_t Maj(uint32_t x, uint32_t y, uint32_t z)
{
	// Section 4.1.2
	return (x & y) ^ (x & z) ^ (y & z);
}

uint32_t SHR(uint32_t x, int n)
{
	// Section 3.2
	return x >> n;
}

uint32_t ROTR(uint32_t x, int n)
{
	// Section 3.2
	return (x >> n) | (x << (32 - n));
}

uint32_t Sig0(uint32_t x)
{
	// Section 4.1.2
	return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

uint32_t Sig1(uint32_t x)
{
	// Section 4.1.2
	return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}

uint32_t sig0(uint32_t x)
{
	// Section 4.1.2
	return ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3);
}

uint32_t sig1(uint32_t x)
{
	// Section 4.1.2
	return ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10);
}

// Return the number of 0 bytes to output
uint8_t nozerobytes(uint8_t noBits)
{
	// ULL means unsigned long long, making sure numbers are treated as 64 bit
	// Amount of bits left for padding
	uint8_t result = 512ULL - (noBits % 512ULL);

	// check if there's enough room to do padding
	if (result < 65) // 65 = 1 bit (required in spec) + 64 bits ()
	{
		// Add 512 to make more space
		result += 512;
	}

	// Why? See video timestamp 34:10, 45:00
	// 72 = 8 bits (1000 0000) + 64 bits (length of file)
	result -= 72;

	return (result / 8ULL);
}

// Read next 512 bit block from file (infile)
// M - 512 bit block to store data
// infile - file to read
// nobits - number of bits currently read from infile
// status - represents state
int nextblock(union block *M, FILE *infile, uint64_t *nobits,
			  enum flag *status)
{
	if (*status == FINISH)
		return 0; // Return finish, don't loop again in main

	if (*status == PAD)
	{
		// Set bits 1 - 448 (56 * 8) to 0
		for (int i = 0; i < 56; i++)
			M->eight[i] = 0;
		// Last 64bit contains the number of bits as per specification
		M->sixtyFour[7] = *nobits;
		*status = FINISH;
		return 1; // Return non-finish int, another round of hashing still has to occur
	}

	// Remaining code only executes in READ state

	// Read 64 1 bytes (512 bytes) from infile to M
	size_t nobytesread = fread(M->eight, 1, 64, infile);
	if (nobytesread == 64) // Has read 512 bits
		return 1;

	// If we can fit all padding in last block since less than 448 bits were read. Same code below as PAD flag
	if (nobytesread < 56)
	{
		// Add 1000 0000
		M->eight[nobytesread] = 0x80;
		for (int i = nobytesread + 1; i < 56; i++)
			M->eight[i] = 0;
		M->sixtyFour[7] = *nobits;
		*status = FINISH;
		return 1;
	}

	// Otherwise we have read between 56 (incl) and 64 (excl) bytes.
	M->eight[nobytesread] = 0x80; // Add 1000 0000 to just after the last byte read
	// Fill remaining block with 0's
	for (int i = nobytesread + 1; i < 64; i++)
		M->eight[i] = 0;
	*status = PAD;
	return 1;
}

// Perform hash calculation on block using initial/previous hash values
void nexthash(union block *M, uint32_t *H)
{
	// Section 6.2.2
	uint32_t W[64];
	uint32_t a, b, c, d, e, f, g, h, T1, T2;
	int t;

	for (t = 0; t < 16; t++)
		W[t] = M->thirtyTwo[t];

	for (t = 16; t < 64; t++)
		W[t] = sig1(W[t - 2]) + W[t - 7] + sig0(W[t - 15]) + W[t - 16];

	a = H[0];
	b = H[1];
	c = H[2];
	d = H[3];
	e = H[4];
	f = H[5];
	g = H[6];
	h = H[7];

	for (t = 0; t < 64; t++)
	{
		T1 = h + Sig1(e) + Ch(e, f, g) + K[t] + W[t];
		T2 = Sig0(a) + Maj(a, b, c);
		h = g;
		g = f;
		f = e;
		e = d + T1;
		d = c;
		c = b;
		b = a;
		a = T1 + T2;
		H[0] = a + H[0];
		H[1] = b + H[1];
		H[2] = c + H[2];
		H[3] = d + H[3];
		H[4] = e + H[4];
		H[5] = f + H[5];
		H[6] = g + H[6];
		H[7] = f + H[7];
	}
}

int main(int argc, char *argv[])
{
	// Check has an argument
	if (argc != 2)
	{
		printf("Error: expected filename as argument.\n");
		return EXIT_FAILURE;
	}

	// Open file as read-only with no newline translation
	FILE *file = fopen(argv[1], "rb");

	if (!file)
	{
		printf("Error: file could not be opened: %s\n", argv[1]);
		return EXIT_FAILURE;
	}

	// Initial hash values as described in Section 5.3.3
	uint32_t H[] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

	// Current padded message block
	union block M;
	uint64_t nobits = 0;
	enum flag status = READ;

	while (nextblock(&M, file, &nobits, &status))
	{
		// Calculate the next hash value
		nexthash(&M, H);
	}

	// Close file
	fclose(file);

	// Output hash
	for (int i = 0; i < 8; i++)
	{
		printf("%02" PRIx32, H[i]);
	}
	printf("\n");

	return EXIT_SUCCESS;
}
