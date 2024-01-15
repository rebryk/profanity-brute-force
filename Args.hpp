#ifndef HPP_ARGS
#define HPP_ARGS

struct Args
{
	bool bHelp = false;
	bool bModeBenchmark = false;
	bool bModeZeros = false;
	bool bModeLetters = false;
	bool bModeNumbers = false;
	bool bModeReverse = false;
	bool bMode16Gb = false;
	bool bModeCache = false;
	bool bModeHashTable = false;
	bool bModeSingle = false;
	int iModeSteps = 0;
	int iModeSkipX = 0;
	int iModeSkipY = 0;
	std::vector< std::string > strModeTarget;
	std::vector< std::string > strModeLeading;
	std::vector< std::string > strModeMatching;
	bool bModeLeadingRange = false;
	bool bModeRange = false;
	bool bModeMirror = false;
	bool bModeDoubles = false;
	int rangeMin = 0;
	int rangeMax = 0;
	std::vector<size_t> vDeviceSkipIndex;
	size_t worksizeLocal = 64;
	size_t worksizeMax = 0; // Will be automatically determined later if not overriden by user
	bool bNoCache = false;
	size_t inverseSize = 255;
	size_t inverseMultiple = 16384;
	bool bMineContract = false;
};

#endif // HPP_ARGS