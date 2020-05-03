// MD5 message-digest algorithm implementation from RFC 1321
// Paul Kenny, G00326057, Software Development Year 4, Galway-Mayo Institute Of Technology

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

// Functions F, G, H, I defined in Section 3.4
#define F(x, y, z) (x & y | ~y & z)
#define G(x, y, z) (x & z | y & ~z)
#define H(x, y, z) (x ^ y ^ z)
#define I(x, y, z) (y ^ (x | ~z))
// Round functions
#define round1(a, b, c, d, k, s, t) (b + ((a + F(b, c, d) + k + T[t]) << s))
#define round2(a, b, c, d, k, s, t) (b + ((a + G(b, c, d) + k + T[t]) << s))
#define round3(a, b, c, d, k, s, t) (b + ((a + H(b, c, d) + k + T[t]) << s))
#define round4(a, b, c, d, k, s, t) (b + ((a + I(b, c, d) + k + T[t]) << s))

// Initial buffer values from Section 3.3
uint32_t buffer[4] = {
	0x01234567,
	0x89abcdef,
	0xfedcba98,
	0x76543210};

// block of data to process
union block {
	uint64_t sixtyFour[8];
	uint32_t thirtyTwo[16];
	uint8_t eight[64];
};

// Precomputed table from https://en.wikipedia.org/wiki/MD5#Pseudocode
uint32_t T[64] = {
	0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
	0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
	0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
	0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
	0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
	0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
	0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
	0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
	0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
	0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
	0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
	0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
	0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
	0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
	0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
	0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

enum flag
{
	READ,
	PAD,
	FINISH
};

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
	// Add to number of bits read
	*nobits = (nobytesread * 8);
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
// See Section 3.4
void nexthash(union block *M)
{
	// Temporary copies
	uint32_t A = buffer[0], B = buffer[1], C = buffer[2], D = buffer[3];

	// Aside: why isn't this in a loop like lots of other implementations? Reasoning is hard.

	/* Round 1. */
	A = round1(A, B, C, D, M->thirtyTwo[0], 7, 0);
	A = round1(D, A, B, C, M->thirtyTwo[1], 12, 1);
	A = round1(C, D, A, B, M->thirtyTwo[2], 17, 2);
	A = round1(B, C, D, A, M->thirtyTwo[3], 22, 3);
	A = round1(A, B, C, D, M->thirtyTwo[4], 7, 4);
	A = round1(D, A, B, C, M->thirtyTwo[5], 12, 5);
	A = round1(C, D, A, B, M->thirtyTwo[6], 17, 6);
	A = round1(B, C, D, A, M->thirtyTwo[7], 22, 7);
	A = round1(A, B, C, D, M->thirtyTwo[8], 7, 8);
	A = round1(D, A, B, C, M->thirtyTwo[9], 12, 9);
	A = round1(C, D, A, B, M->thirtyTwo[10], 17, 10);
	A = round1(B, C, D, A, M->thirtyTwo[11], 22, 11);
	A = round1(A, B, C, D, M->thirtyTwo[12], 7, 12);
	A = round1(D, A, B, C, M->thirtyTwo[13], 12, 13);
	A = round1(C, D, A, B, M->thirtyTwo[14], 17, 14);
	A = round1(B, C, D, A, M->thirtyTwo[15], 22, 15);

	/* Round 2. */
	A = round2(A, B, C, D, M->thirtyTwo[1], 5, 16);
	A = round2(D, A, B, C, M->thirtyTwo[6], 9, 17);
	A = round2(C, D, A, B, M->thirtyTwo[11], 14, 18);
	A = round2(B, C, D, A, M->thirtyTwo[0], 20, 19);
	A = round2(A, B, C, D, M->thirtyTwo[5], 5, 20);
	A = round2(D, A, B, C, M->thirtyTwo[10], 9, 21);
	A = round2(C, D, A, B, M->thirtyTwo[15], 14, 22);
	A = round2(B, C, D, A, M->thirtyTwo[4], 20, 23);
	A = round2(A, B, C, D, M->thirtyTwo[9], 5, 24);
	A = round2(D, A, B, C, M->thirtyTwo[14], 9, 25);
	A = round2(C, D, A, B, M->thirtyTwo[3], 14, 26);
	A = round2(B, C, D, A, M->thirtyTwo[8], 20, 27);
	A = round2(A, B, C, D, M->thirtyTwo[13], 5, 28);
	A = round2(D, A, B, C, M->thirtyTwo[2], 9, 29);
	A = round2(C, D, A, B, M->thirtyTwo[7], 14, 30);
	A = round2(B, C, D, A, M->thirtyTwo[12], 20, 31);

	/* Round 3. */
	A = round3(A, B, C, D, M->thirtyTwo[5], 4, 32);
	A = round3(D, A, B, C, M->thirtyTwo[8], 11, 33);
	A = round3(C, D, A, B, M->thirtyTwo[11], 16, 34);
	A = round3(B, C, D, A, M->thirtyTwo[14], 23, 35);
	A = round3(A, B, C, D, M->thirtyTwo[1], 4, 36);
	A = round3(D, A, B, C, M->thirtyTwo[4], 11, 37);
	A = round3(C, D, A, B, M->thirtyTwo[7], 16, 38);
	A = round3(B, C, D, A, M->thirtyTwo[10], 23, 39);
	A = round3(A, B, C, D, M->thirtyTwo[13], 4, 40);
	A = round3(D, A, B, C, M->thirtyTwo[0], 11, 41);
	A = round3(C, D, A, B, M->thirtyTwo[3], 16, 42);
	A = round3(B, C, D, A, M->thirtyTwo[6], 23, 43);
	A = round3(A, B, C, D, M->thirtyTwo[9], 4, 44);
	A = round3(D, A, B, C, M->thirtyTwo[12], 11, 45);
	A = round3(C, D, A, B, M->thirtyTwo[15], 16, 46);
	A = round3(B, C, D, A, M->thirtyTwo[2], 23, 47);

	/* Round 4. */
	A = round4(A, B, C, D, M->thirtyTwo[0], 6, 48);
	A = round4(D, A, B, C, M->thirtyTwo[7], 10, 49);
	A = round4(C, D, A, B, M->thirtyTwo[14], 15, 50);
	A = round4(B, C, D, A, M->thirtyTwo[5], 21, 51);
	A = round4(A, B, C, D, M->thirtyTwo[12], 6, 52);
	A = round4(D, A, B, C, M->thirtyTwo[3], 10, 53);
	A = round4(C, D, A, B, M->thirtyTwo[10], 15, 54);
	A = round4(B, C, D, A, M->thirtyTwo[1], 21, 55);
	A = round4(A, B, C, D, M->thirtyTwo[8], 6, 56);
	A = round4(D, A, B, C, M->thirtyTwo[15], 10, 57);
	A = round4(C, D, A, B, M->thirtyTwo[6], 15, 58);
	A = round4(B, C, D, A, M->thirtyTwo[13], 21, 59);
	A = round4(A, B, C, D, M->thirtyTwo[4], 6, 60);
	A = round4(D, A, B, C, M->thirtyTwo[11], 10, 61);
	A = round4(C, D, A, B, M->thirtyTwo[2], 15, 62);
	A = round4(B, C, D, A, M->thirtyTwo[9], 21, 63);

	buffer[0] += A;
	buffer[1] += B;
	buffer[2] += C;
	buffer[3] += D;
}

void md5(char *filename)
{
	// Open file as read-only with no newline translation
	// "rb" means read-only, binary
	FILE *file = fopen(filename, "rb");

	if (!file)
	{
		printf("Error: file could not be opened: %s\nEnter argument --help for usage information\n", filename);
		return EXIT_FAILURE;
	}

	// Current padded message block
	union block M;
	uint64_t nobits = 0;
	enum flag status = READ;

	// Read next 512 bit block from file, then calculate hash
	while (nextblock(&M, file, &nobits, &status))
	{
		// Calculate the next hash value
		nexthash(&M);
	}

	// Close file
	fclose(file);

	// Output hash
	for (int i = 0; i < 4; i++)
	{
		printf("%02" PRIX32, buffer[i]);
	}
	printf("\n");
}

// Print help information and exit program
void printHelpAndExit()
{
	printf(
		"Usage: md5 [options] [file]\n"
		"  Options:\n"
		"    --help Display help information\n"
		"    --test Perform self tests\n"
		"  Examples:\n"
		"    md5 /path/to/file\n"
		"    md5 --help\n"
		"    md5 --test\n"
	);
	exit(EXIT_SUCCESS);
}

// Perform self tests on MD5 algorithm
void performTests()
{
	// tests unimplemented
}

int main(int argc, char *argv[])
{
	// Check has an argument
	if (argc != 2)
	{
		printf("Error: expected one argument.\nEnter argument --help for usage information\n");
		return EXIT_FAILURE;
	}

	// Handle program arguments
	if (strcmp(argv[1], "--help") == 0)
	{
		printHelpAndExit();
	}
	else if (strcmp(argv[1], "--test") == 0)
	{
		performTests();
	}

	md5(argv[1]);

	return 0;
}
