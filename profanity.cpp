#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <map>
#include <set>
#include <thread>

#define CL_TARGET_OPENCL_VERSION 300

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h> // Included to get topology to get an actual unique identifier per device
#else
#include <CL/cl.h>
#include <CL/cl_ext.h> // Included to get topology to get an actual unique identifier per device
#endif

#define CL_DEVICE_PCI_BUS_ID_NV  0x4008
#define CL_DEVICE_PCI_SLOT_ID_NV 0x4009

#include "Dispatcher.hpp"
#include "ArgParser.hpp"
#include "Args.hpp"
#include "Mode.hpp"
#include "help.hpp"

std::string readFile(const char * const szFilename)
{
	std::ifstream in(szFilename, std::ios::in | std::ios::binary);
	std::ostringstream contents;
	contents << in.rdbuf();
	return contents.str();
}

std::vector<cl_device_id> getAllDevices(cl_device_type deviceType = CL_DEVICE_TYPE_GPU)
{
	std::vector<cl_device_id> vDevices;

	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, NULL, &platformIdCount);

	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), NULL);

	for( auto it = platformIds.cbegin(); it != platformIds.cend(); ++it ) {
		cl_uint countDevice;
		clGetDeviceIDs(*it, deviceType, 0, NULL, &countDevice);

		std::vector<cl_device_id> deviceIds(countDevice);
		clGetDeviceIDs(*it, deviceType, countDevice, deviceIds.data(), &countDevice);

		std::copy( deviceIds.begin(), deviceIds.end(), std::back_inserter(vDevices) );
	}

	return vDevices;
}

template <typename T, typename U, typename V, typename W>
T clGetWrapper(U function, V param, W param2) {
	T t;
	function(param, param2, sizeof(t), &t, NULL);
	return t;
}

template <typename U, typename V, typename W>
std::string clGetWrapperString(U function, V param, W param2) {
	size_t len;
	function(param, param2, 0, NULL, &len);
	char * const szString = new char[len];
	function(param, param2, len, szString, NULL);
	std::string r(szString);
	delete[] szString;
	return r;
}

template <typename T, typename U, typename V, typename W>
std::vector<T> clGetWrapperVector(U function, V param, W param2) {
	size_t len;
	function(param, param2, 0, NULL, &len);
	len /= sizeof(T);
	std::vector<T> v;
	if (len > 0) {
		T * pArray = new T[len];
		function(param, param2, len * sizeof(T), pArray, NULL);
		for (size_t i = 0; i < len; ++i) {
			v.push_back(pArray[i]);
		}
		delete[] pArray;
	}
	return v;
}

std::vector<std::string> getBinaries(cl_program & clProgram) {
	std::vector<std::string> vReturn;
	auto vSizes = clGetWrapperVector<size_t>(clGetProgramInfo, clProgram, CL_PROGRAM_BINARY_SIZES);
	if (!vSizes.empty()) {
		unsigned char * * pBuffers = new unsigned char *[vSizes.size()];
		for (size_t i = 0; i < vSizes.size(); ++i) {
			pBuffers[i] = new unsigned char[vSizes[i]];
		}

		clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, vSizes.size() * sizeof(unsigned char *), pBuffers, NULL);
		for (size_t i = 0; i < vSizes.size(); ++i) {
			std::string strData(reinterpret_cast<char *>(pBuffers[i]), vSizes[i]);
			vReturn.push_back(strData);
			delete[] pBuffers[i];
		}

		delete[] pBuffers;
	}

	return vReturn;
}

unsigned int getUniqueDeviceIdentifier(const cl_device_id & deviceId) {
#if defined(CL_DEVICE_TOPOLOGY_AMD)
	auto topology = clGetWrapper<cl_device_topology_amd>(clGetDeviceInfo, deviceId, CL_DEVICE_TOPOLOGY_AMD);
	if (topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {
		return (topology.pcie.bus << 16) + (topology.pcie.device << 8) + topology.pcie.function;
	}
#endif
	cl_int bus_id = clGetWrapper<cl_int>(clGetDeviceInfo, deviceId, CL_DEVICE_PCI_BUS_ID_NV);
	cl_int slot_id = clGetWrapper<cl_int>(clGetDeviceInfo, deviceId, CL_DEVICE_PCI_SLOT_ID_NV);
	return (bus_id << 16) + slot_id;
}

template <typename T> bool printResult(const T & t, const cl_int & err) {
	std::cout << ((t == NULL) ? toString(err) : "OK") << std::endl;
	return t == NULL;
}

bool printResult(const cl_int err) {
	std::cout << ((err != CL_SUCCESS) ? toString(err) : "OK") << std::endl;
	return err != CL_SUCCESS;
}

std::string getDeviceCacheFilename(cl_device_id & d, const size_t & inverseSize) {
	const auto uniqueId = getUniqueDeviceIdentifier(d);
	return "cache-opencl." + toString(inverseSize) + "." + toString(uniqueId);
}

bool detectMode(Args const & args, Mode & mode, std::string const & str)
{
	if (args.bModeBenchmark) {
		mode = Mode::benchmark();
	} else if (args.bModeZeros) {
		mode = Mode::zeros();
	} else if (args.bModeLetters) {
		mode = Mode::letters();
	} else if (args.bModeNumbers) {
		mode = Mode::numbers();
	} else if (!args.strModeLeading.empty()) {
		mode = Mode::leading(str.front());
	} else if (!args.strModeMatching.empty()) {
		mode = Mode::matching(str);
	} else if (args.bModeLeadingRange) {
		mode = Mode::leadingRange(args.rangeMin, args.rangeMax);
	} else if (args.bModeRange) {
		mode = Mode::range(args.rangeMin, args.rangeMax);
	} else if(args.bModeMirror) {
		mode = Mode::mirror();
	} else if (args.bModeDoubles) {
		mode = Mode::doubles();
	} else if (args.bModeHashTable) {
		mode = Mode::hashTable(args.bMode16Gb, args.iModeSkipY);
	} else if (args.bModeReverse) {
		if (str.empty()) {
			std::cout << "Specify a target public address with -t" << std::endl;
			return false;
		}

		if (str.size() != 130) {
			std::cout << "Target public address must be 130 characters long and start with 0x" << std::endl;
			return false;
		}

		if (args.iModeSteps == 0) {
			std::cout << "Specify the number of steps with -s" << std::endl;
			return false;
		}

		mode = Mode::reverse(
				str
			, 	args.iModeSteps
			, 	args.bMode16Gb
			, 	args.bModeCache
			, 	args.iModeSkipX
			, 	args.iModeSkipY
			, 	args.bModeSingle
		);
	}
	else {
		std::cout << g_strHelp << std::endl;
		return false;
	}
	std::cout << "Mode: " << mode.name << std::endl;

	if (args.bMineContract) {
		mode.target = CONTRACT;
	} else {
		mode.target = ADDRESS;
	}
	std::cout << "Target: " << mode.transformName() << std:: endl;

	return true;
}

bool process(Args const & args, std::string const & str)
{
	std::cout << "TEST: " << str << std::endl;
	Mode mode = Mode::benchmark();

	if (!detectMode(args, mode, str))
		return false;

	std::vector<cl_device_id> vFoundDevices = getAllDevices();
	std::vector<cl_device_id> vDevices;
	std::map<cl_device_id, size_t> mDeviceIndex;

	std::vector<std::string> vDeviceBinary;
	std::vector<size_t> vDeviceBinarySize;
	cl_int errorCode;
	bool bUsedCache = false;

	std::cout << "Devices:" << std::endl;
	for (size_t i = 0; i < vFoundDevices.size(); ++i) {
		// Ignore devices in skip index
		if (std::find(args.vDeviceSkipIndex.begin(), args.vDeviceSkipIndex.end(), i) != args.vDeviceSkipIndex.end()) {
			continue;
		}

		cl_device_id & deviceId = vFoundDevices[i];

		const auto strName = clGetWrapperString(clGetDeviceInfo, deviceId, CL_DEVICE_NAME);
		const auto computeUnits = clGetWrapper<cl_uint>(clGetDeviceInfo, deviceId, CL_DEVICE_MAX_COMPUTE_UNITS);
		const auto globalMemSize = clGetWrapper<cl_ulong>(clGetDeviceInfo, deviceId, CL_DEVICE_GLOBAL_MEM_SIZE);
		bool precompiled = false;

		// Check if there's a prebuilt binary for this device and load it
		if(!args.bNoCache) {
			std::ifstream fileIn(getDeviceCacheFilename(deviceId, args.inverseSize), std::ios::binary);
			if (fileIn.is_open()) {
				vDeviceBinary.push_back(std::string((std::istreambuf_iterator<char>(fileIn)), std::istreambuf_iterator<char>()));
				vDeviceBinarySize.push_back(vDeviceBinary.back().size());
				precompiled = true;
			}
		}

		std::cout << "  GPU" << i << ": " << strName << ", " << globalMemSize << " bytes available, " << computeUnits << " compute units (precompiled = " << (precompiled ? "yes" : "no") << ")" << std::endl;
		vDevices.push_back(vFoundDevices[i]);
		mDeviceIndex[vFoundDevices[i]] = i;
	}

	if (vDevices.empty()) {
		return false;
	}

	std::cout << std::endl;
	std::cout << "Initializing OpenCL..." << std::endl;
	std::cout << "  Creating context..." << std::flush;
	auto clContext = clCreateContext( NULL, vDevices.size(), vDevices.data(), NULL, NULL, &errorCode);
	if (printResult(clContext, errorCode)) {
		return false;
	}

	cl_program clProgram;
	if (vDeviceBinary.size() == vDevices.size()) {
		// Create program from binaries
		bUsedCache = true;

		std::cout << "  Loading kernel from binary..." << std::flush;
		const unsigned char * * pKernels = new const unsigned char *[vDevices.size()];
		for (size_t i = 0; i < vDeviceBinary.size(); ++i) {
			pKernels[i] = reinterpret_cast<const unsigned char *>(vDeviceBinary[i].data());
		}

		cl_int * pStatus = new cl_int[vDevices.size()];

		clProgram = clCreateProgramWithBinary(clContext, vDevices.size(), vDevices.data(), vDeviceBinarySize.data(), pKernels, pStatus, &errorCode);
		if(printResult(clProgram, errorCode)) {
			return 1;
		}
	} else {
		// Create a program from the kernel source
		std::cout << "  Compiling kernel..." << std::flush;
		const std::string strKeccak = readFile("keccak.cl");
		const std::string strVanity = readFile("profanity.cl");
		const char * szKernels[] = { strKeccak.c_str(), strVanity.c_str() };

		clProgram = clCreateProgramWithSource(clContext, sizeof(szKernels) / sizeof(char *), szKernels, NULL, &errorCode);
		if (printResult(clProgram, errorCode)) {
			return false;
		}
	}

	// Build the program
	std::cout << "  Building program..." << std::flush;
	const std::string strBuildOptions = "-D PROFANITY_INVERSE_SIZE=" + toString(args.inverseSize) + " -D PROFANITY_MAX_SCORE=" + toString(PROFANITY_MAX_SCORE);
	if (printResult(clBuildProgram(clProgram, vDevices.size(), vDevices.data(), strBuildOptions.c_str(), NULL, NULL))) {
#ifdef PROFANITY_DEBUG
		std::cout << std::endl;
		std::cout << "build log:" << std::endl;

		size_t sizeLog;
		clGetProgramBuildInfo(clProgram, vDevices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &sizeLog);
		char * const szLog = new char[sizeLog];
		clGetProgramBuildInfo(clProgram, vDevices[0], CL_PROGRAM_BUILD_LOG, sizeLog, szLog, NULL);

		std::cout << szLog << std::endl;
		delete[] szLog;
#endif
		return false;
	}

	// Save binary to improve future start times
	if( !bUsedCache && !args.bNoCache ) {
		std::cout << "  Saving program..." << std::flush;
		auto binaries = getBinaries(clProgram);
		for (size_t i = 0; i < binaries.size(); ++i) {
			std::ofstream fileOut(getDeviceCacheFilename(vDevices[i], args.inverseSize), std::ios::binary);
			fileOut.write(binaries[i].data(), binaries[i].size());
		}
		std::cout << "OK" << std::endl;
	}

	std::cout << std::endl;
	Dispatcher d(
			clContext
		, 	clProgram
		, 	mode
		, 	args.worksizeMax == 0 ? args.inverseSize * args.inverseMultiple : args.worksizeMax
		, 	args.inverseSize
		, 	args.inverseMultiple
		, 	0
	);

	for (auto & i : vDevices) {
		d.addDevice(i, args.worksizeLocal, mDeviceIndex[i]);
	}
		
	d.run();

	clReleaseContext(clContext);

	return true;
}

int main(int argc, char * * argv) {
	try {
		ArgParser argp(argc, argv);
		Args args;

		argp.addSwitch('h', "help", args.bHelp);
		argp.addSwitch('0', "benchmark", args.bModeBenchmark);
		argp.addSwitch('1', "zeros", args.bModeZeros);
		argp.addSwitch('2', "letters", args.bModeLetters);
		argp.addSwitch('3', "numbers", args.bModeNumbers);
		argp.addMultiSwitch('4', "leading", args.strModeLeading);
		argp.addMultiSwitch('5', "matching", args.strModeMatching);
		argp.addSwitch('6', "leading-range", args.bModeLeadingRange);
		argp.addSwitch('7', "range", args.bModeRange);
		argp.addSwitch('8', "mirror", args.bModeMirror);
		argp.addSwitch('9', "leading-doubles", args.bModeDoubles);
		argp.addSwitch('m', "min", args.rangeMin);
		argp.addSwitch('M', "max", args.rangeMax);
		argp.addMultiSwitch('s', "skip", args.vDeviceSkipIndex);
		argp.addSwitch('w', "work", args.worksizeLocal);
		argp.addSwitch('W', "work-max", args.worksizeMax);
		argp.addSwitch('n', "no-cache", args.bNoCache);
		argp.addSwitch('i', "inverse-size", args.inverseSize);
		argp.addSwitch('I', "inverse-multiple", args.inverseMultiple);
		argp.addSwitch('c', "contract", args.bMineContract);
		argp.addSwitch('r', "reverse", args.bModeReverse);
		argp.addMultiSwitch('t', "target", args.strModeTarget);
		argp.addSwitch('e', "extended", args.bMode16Gb);
		argp.addSwitch('C', "cache", args.bModeCache);
		argp.addSwitch('s', "steps", args.iModeSteps);
		argp.addSwitch('x', "skip-x", args.iModeSkipX);
		argp.addSwitch('y', "skip-y", args.iModeSkipY);
		argp.addSwitch('h', "hash-table", args.bModeHashTable);
		argp.addSwitch('S', "single", args.bModeSingle);

		if (!argp.parse()) {
			std::cout << "error: bad arguments, try again :<" << std::endl;
			return 1;
		}

		if (args.bHelp) {
			std::cout << g_strHelp << std::endl;
			return 0;
		}

		auto const & strVec = args.strModeLeading.empty() ? 
			(args.strModeMatching.empty() ? args.strModeTarget : args.strModeMatching) :
		args.strModeLeading;

		for (auto const & str : strVec)
		{
			std::thread th(process, args, str);
			
			th.join();
		}

		return 0;
	} catch (std::runtime_error & e) {
		std::cout << "std::runtime_error - " << e.what() << std::endl;
	} catch (...) {
		std::cout << "unknown exception occured" << std::endl;
	}

	return 1;
}

