#include "Dispatcher.hpp"

// Includes
#include <stdexcept>
#include <iostream>
#include <thread>
#include <sstream>
#include <iomanip>
#include <random>
#include <thread>
#include <algorithm>
#include <climits>
#include <cassert>

#include "precomp.hpp"

static const size_t STEPS_OFFSET = 3;
static const size_t BATCH_X_SIZE = 2000; // num steps per batch
static const size_t HASH_TABLE_SIZE = 1U << 30; // 1U << 30;
static const size_t HASH_TABLE_SET_SIZE = 1 << 25; // 1 << 25;
static const size_t HASH_TABLE_JOB_SIZE = 1 << 18; // 1 << 18;
static const size_t HASH_TABLE_LOAD_SIZE = 1 << 18; // 1 << 18;

static std::string toHex(const uint8_t * const s, const size_t len) {
	std::string b("0123456789abcdef");
	std::string r;

	for (size_t i = 0; i < len; ++i) {
		const unsigned char h = s[i] / 16;
		const unsigned char l = s[i] % 16;

		r = r + b.substr(h, 1) + b.substr(l, 1);
	}

	return r;
}

cl_ulong4 restorePrivateKey(cl_ulong4 seed, cl_uint id, cl_ulong round) {
	cl_ulong4 privateKey;
	cl_ulong carry = 0;
	privateKey.s[0] = seed.s[0] + round; carry = privateKey.s[0] < round;
	privateKey.s[1] = seed.s[1] + carry; carry = !privateKey.s[1];
	privateKey.s[2] = seed.s[2] + carry; carry = !privateKey.s[2];
	privateKey.s[3] = seed.s[3] + carry + id;
	return privateKey;
}

std::string privateKeyToStr(cl_ulong4 privateKey) {
	std::ostringstream ss;
	ss << std::hex << std::setfill('0');
	ss << std::setw(16) << privateKey.s[3] << std::setw(16) << privateKey.s[2] << std::setw(16) << privateKey.s[1] << std::setw(16) << privateKey.s[0];
	return ss.str();
}

static void printResult(cl_ulong4 seed, cl_ulong round, result r, cl_uchar score, const std::chrono::time_point<std::chrono::steady_clock> & timeStart, const Mode & mode) {
	// Time delta
	const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart).count();

	// Format private key
	cl_ulong4 privateKey = restorePrivateKey(seed, r.foundId, round);
	const std::string strPrivate = privateKeyToStr(privateKey);

	// Format public key
	const std::string strPublic = toHex(r.foundHash, 20);

	// Print
	const std::string strVT100ClearLine = "\33[2K\r";
	std::cout << strVT100ClearLine << "  Time: " << std::setw(5) << seconds << "s Score: " << std::setw(2) << (int) score << " Private: 0x" << strPrivate << ' ';

	std::cout << mode.transformName();
	std::cout << ": 0x" << strPublic << " id: " << r.foundId << " round: " << int(round) << std::endl;
}

unsigned int getKernelExecutionTimeMicros(cl_event & e) {
	cl_ulong timeStart = 0, timeEnd = 0;
	clWaitForEvents(1, &e);
	clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
	clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);
	return (timeEnd - timeStart) / 1000;
}

Dispatcher::OpenCLException::OpenCLException(const std::string s, const cl_int res) :
	std::runtime_error( s + " (res = " + toString(res) + ")"),
	m_res(res)
{

}

void Dispatcher::OpenCLException::OpenCLException::throwIfError(const std::string s, const cl_int res) {
	if (res != CL_SUCCESS) {
		throw OpenCLException(s, res);
	}
}

cl_command_queue Dispatcher::Device::createQueue(cl_context & clContext, cl_device_id & clDeviceId) {
	// nVidia CUDA Toolkit 10.1 only supports OpenCL 1.2 so we revert back to older functions for compatability
#ifdef PROFANITY_DEBUG
	cl_command_queue_properties p = CL_QUEUE_PROFILING_ENABLE;
#else
	cl_command_queue_properties p = NULL;
#endif

#ifdef CL_VERSION_2_0
	const cl_command_queue ret = clCreateCommandQueueWithProperties(clContext, clDeviceId, &p, NULL);
#else
	const cl_command_queue ret = clCreateCommandQueue(clContext, clDeviceId, p, NULL);
#endif
	return ret == NULL ? throw std::runtime_error("failed to create command queue") : ret;
}

cl_kernel Dispatcher::Device::createKernel(cl_program & clProgram, const std::string s) {
	cl_kernel ret  = clCreateKernel(clProgram, s.c_str(), NULL);
	return ret == NULL ? throw std::runtime_error("failed to create kernel \"" + s + "\"") : ret;
}

cl_ulong4 Dispatcher::Device::createSeed() {
#ifdef PROFANITY_DEBUG
	cl_ulong4 r;
	r.s[0] = 1;
	r.s[1] = 1;
	r.s[2] = 1;
	r.s[3] = 1;
	return r;
#else
	// Randomize private keys
	std::random_device rd;
	uint seed = rd();
	seed = (1 << 18) + (seed) % (1 << 18);
	std::mt19937_64 eng(seed);
	std::uniform_int_distribution<cl_ulong> distr;

	cl_ulong4 r;
	r.s[0] = distr(eng);
	r.s[1] = distr(eng);
	r.s[2] = distr(eng);
	r.s[3] = distr(eng);
	return r;
#endif
}

Dispatcher::Device::Device(Dispatcher & parent, cl_context & clContext, cl_program & clProgram, cl_device_id clDeviceId, const size_t worksizeLocal, const size_t size, const size_t index, const Mode & mode) :
	m_parent(parent),
	m_index(index),
	m_clDeviceId(clDeviceId),
	m_worksizeLocal(worksizeLocal),
	m_clScoreMax(0),
	m_clQueue(createQueue(clContext, clDeviceId) ),
	m_kernelInit(createKernel(clProgram, mode.name == "reverse" ? "profanity_init_reverse" : "profanity_init")),
	m_kernelInitHashTable(createKernel(clProgram, mode.cache ? "profanity_init_hash_table_from_bytes" : "profanity_init_hash_table")),
	m_kernelInverse(createKernel(clProgram, mode.name == "reverse" ? "profanity_inverse_reverse" : "profanity_inverse")),
	m_kernelIterate(createKernel(clProgram, mode.name == "reverse" ? "profanity_iterate_reverse" : "profanity_iterate")),
	m_kernelTransform( mode.transformKernel() == "" ? NULL : createKernel(clProgram, mode.transformKernel())),
	m_kernelClearResults(createKernel(clProgram, "profanity_clear_results")),
	m_kernelScore(createKernel(clProgram, mode.kernel)),
	m_memPrecomp(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(g_precomp), g_precomp),
	m_memPointsDeltaX(clContext, m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size, true),
	m_memInversedNegativeDoubleGy(clContext, m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size),
	m_memPrevLambda(clContext, m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size, true),
	m_memResult(clContext, m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, PROFANITY_MAX_SCORE + 1),
	m_memData1(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 20),
	m_memData2(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 20),
	m_memHashTable(clContext,  m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 2 * HASH_TABLE_SIZE * (mode.extended ? 2 : 1), !(mode.name == "reverse" || mode.name == "hashTable")),
	m_memSeed(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, HASH_TABLE_JOB_SIZE, !((mode.name == "reverse" && !mode.cache) || mode.name == "hashTable")),
	m_memPublicAddress(clContext, m_clQueue, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 3 * HASH_TABLE_JOB_SIZE, !(mode.name == "reverse" || mode.name == "hashTable")),
	m_memPublicBytes(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 3 * HASH_TABLE_LOAD_SIZE, !(mode.name == "reverse" && mode.cache)),
	m_clSeed(createSeed()),
	m_round(0),
	m_speed(PROFANITY_SPEEDSAMPLES),
	m_sizeInitialized(0),
	m_sizeHashTableInitialized(0),
	m_eventFinished(NULL),
	m_mode(mode),
	m_batchIndex(0)
{

}

Dispatcher::Device::~Device() {

}

Dispatcher::Dispatcher(cl_context & clContext, cl_program & clProgram, const Mode mode, const size_t worksizeMax, const size_t inverseSize, const size_t inverseMultiple, const cl_uchar clScoreQuit)
	: m_clContext(clContext), m_clProgram(clProgram), m_mode(mode), m_worksizeMax(worksizeMax), m_inverseSize(inverseSize), m_size(inverseSize*inverseMultiple), m_HashTableSize(HASH_TABLE_SET_SIZE * (mode.extended ? 2 : 1)), m_clScoreMax(mode.score), m_clScoreQuit(clScoreQuit), m_eventFinished(NULL), m_countPrint(0) {

}

Dispatcher::~Dispatcher() {

}

void Dispatcher::addDevice(cl_device_id clDeviceId, const size_t worksizeLocal, const size_t index) {
	Device * pDevice = new Device(*this, m_clContext, m_clProgram, clDeviceId, worksizeLocal, m_size, index, m_mode);
	m_vDevices.push_back(pDevice);
}

void printHexNumber(const mp_number & number) {
	for (size_t i = 0; i < 8; ++i) {
		std::cout << std::hex << std::setw(2) << std::setfill('0') << "0x" << number.d[i];
		if (i != 7) {
			std::cout << ", ";
		}
	}
	std::cout << std::dec;
}

void printTargetAddress(const point& target) {
	std::cout << "Target public address:" << std::endl;
	
	std::cout << "x = {{";
	printHexNumber(target.x);
	std::cout << "}}" << std::endl;

	std::cout << "y = {{";
	printHexNumber(target.y);
	std::cout << "}}" << std::endl;
}

void Dispatcher::runReverse() {
	const auto isReverse = m_mode.name == "reverse";

	const size_t m_batchXTotal = isReverse ? ((m_mode.steps + BATCH_X_SIZE - 1) / BATCH_X_SIZE) : 1;

	const int nJobs = m_mode.extended ? 64 : 128;
	const size_t m_batchYTotal = nJobs / m_vDevices.size();

	std::cout << "Memory limit: " << (m_mode.extended ? "16Gb" : "8Gb") << std::endl;
	std::cout << "Number of batches: " << m_batchXTotal * m_batchYTotal << std::endl;
	std::cout << "Number of X batches: " << m_batchXTotal << std::endl;
	std::cout << "Number of Y batches: " << m_batchYTotal << std::endl;

	if (isReverse) {
		std::cout << "Number of steps: " << m_mode.steps << std::endl;
		printTargetAddress(m_mode.targetAddress);
	}

	if (m_mode.skipX) {
		std::cout << "Skip x: " << m_mode.skipX << " batches" << std::endl;
	}

	if (m_mode.skipY) {
		std::cout << "Skip y: " << m_mode.skipY << " batches" << std::endl;
	}

	timeStart = std::chrono::steady_clock::now();
	for (m_batchX = m_mode.skipX; m_batchX < m_batchXTotal && m_clScoreMax != PROFANITY_MAX_SCORE; ++m_batchX) {
		for (m_batchY = (m_batchX == m_mode.skipX ? m_mode.skipY : 0); m_batchY < m_batchYTotal && m_clScoreMax != PROFANITY_MAX_SCORE; ++m_batchY) {
			m_quit = false;
			m_countRunning = m_vDevices.size();
			for (size_t i = 0; i < m_countRunning; i++) {
				m_vDevices[i]->m_batchIndex = m_batchY * m_countRunning + i;
			}

			std::cout << "Run batch x=" << m_batchX + 1 << " y=" << m_batchY + 1 << "..." << std::endl;
			std::cout << "Run initialization..." << std::endl;
			timeInitStart = std::chrono::steady_clock::now();
			init();
			const auto timeInitialization = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeInitStart).count();
			std::cout << "Initialization time: " << timeInitialization << " seconds" << std::endl;
			
			if (!isReverse) {
				continue;
			}

			std::cout << "Run search..." << std::endl;
			timeRunStart = std::chrono::steady_clock::now();
			m_eventFinished = clCreateUserEvent(m_clContext, NULL);

			for (auto it = m_vDevices.begin(); it != m_vDevices.end(); ++it) {
				dispatch(*(*it));
			}

			clWaitForEvents(1, &m_eventFinished);
			clReleaseEvent(m_eventFinished);
			m_eventFinished = NULL;

			if (!m_quit) {
				const auto timeRun = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeRunStart).count();
				std::cout << "Search time: " << timeRun << " seconds" << std::endl;
			}

			// Break if single mode is enabled
			if (m_mode.single) {
				m_clScoreMax = PROFANITY_MAX_SCORE;
			}
		}
	}
}

void Dispatcher::run() {
	if (m_mode.name == "reverse" || m_mode.name == "hashTable") {
		runReverse();
		return;
	}

	m_eventFinished = clCreateUserEvent(m_clContext, NULL);
	timeStart = std::chrono::steady_clock::now();

	init();

	const auto timeInitialization = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart).count();
	std::cout << "Initialization time: " << timeInitialization << " seconds" << std::endl;

	m_quit = false;
	m_countRunning = m_vDevices.size();

	std::cout << "Running..." << std::endl;
	std::cout << "  Always verify that a private key generated by this program corresponds to the" << std::endl;
	std::cout << "  public key printed by importing it to a wallet of your choice. This program" << std::endl;
	std::cout << "  like any software might contain bugs and it does by design cut corners to" << std::endl;
	std::cout << "  improve overall performance." << std::endl;
	std::cout << std::endl;

	for (auto it = m_vDevices.begin(); it != m_vDevices.end(); ++it) {
		dispatch(*(*it));
	}

	clWaitForEvents(1, &m_eventFinished);
	clReleaseEvent(m_eventFinished);
	m_eventFinished = NULL;
}

void Dispatcher::init() {
	if (m_mode.name != "reverse" && m_mode.name != "hashTable") {
		std::cout << "Initializing devices..." << std::endl;
		std::cout << "  This should take less than a minute. The number of objects initialized on each" << std::endl;
		std::cout << "  device is equal to inverse-size * inverse-multiple. To lower" << std::endl;
		std::cout << "  initialization time (and memory footprint) I suggest lowering the" << std::endl;
		std::cout << "  inverse-multiple first. You can do this via the -I switch. Do note that" << std::endl;
		std::cout << "  this might negatively impact your performance." << std::endl;
		std::cout << std::endl;
	}

	const auto deviceCount = m_vDevices.size();
	m_sizeInitTotal = m_size * deviceCount;
	m_sizeInitDone = 0;

	m_sizeHashTableInitTotal = m_HashTableSize * deviceCount;
	m_sizeHashTableInitDone = 0;

	m_stepsTotal = BATCH_X_SIZE * deviceCount;
	m_stepsDone = 0;

	cl_event * const pInitEvents = new cl_event[deviceCount];

	for (size_t i = 0; i < deviceCount; ++i) {
		pInitEvents[i] = clCreateUserEvent(m_clContext, NULL);
		m_vDevices[i]->m_eventFinished = pInitEvents[i];
		initBegin(*m_vDevices[i]);
	}

	clWaitForEvents(deviceCount, pInitEvents);
	for (size_t i = 0; i < deviceCount; ++i) {
		m_vDevices[i]->m_eventFinished = NULL;
		clReleaseEvent(pInitEvents[i]);
	}

	delete[] pInitEvents;

	std::cout << std::endl;
}

void Dispatcher::initBegin(Device & d) {
	d.m_round = 0;
	d.m_sizeInitialized = 0;
	d.m_sizeHashTableInitialized = 0;
	d.m_iterHashTableInitialized = 0;
	
	// Reset the hash table
	// d.m_addressToIndex.clear();
	// d.m_addressToIndex.reserve(m_HashTableSize);
	
	// Initialize the list with addresses
	d.m_addresses.clear();
	d.m_addresses.resize(m_HashTableSize);

	// Set mode data
	for (auto i = 0; i < 20; ++i) {
		d.m_memData1[i] = m_mode.data1[i];
		d.m_memData2[i] = m_mode.data2[i];
	}

	// Write precompute table and mode data
	d.m_memPrecomp.write(true);
	d.m_memData1.write(true);
	d.m_memData2.write(true);

	// Kernel arguments - profanity_begin
	d.m_memPrecomp.setKernelArg(d.m_kernelInit, 0);
	d.m_memPointsDeltaX.setKernelArg(d.m_kernelInit, 1);
	d.m_memPrevLambda.setKernelArg(d.m_kernelInit, 2);
	d.m_memResult.setKernelArg(d.m_kernelInit, 3);

	if (m_mode.name == "reverse") {
		CLMemory<point>::setKernelArg(d.m_kernelInit, 4, m_mode.targetAddress);
		CLMemory<cl_ulong>::setKernelArg(d.m_kernelInit, 5, BATCH_X_SIZE * m_batchX);
	} else {
		CLMemory<cl_ulong4>::setKernelArg(d.m_kernelInit, 4, d.m_clSeed);
	}
	
	// Kernel arguments - profanity_init_hash_table
	if (m_mode.cache) {
		d.m_memPublicBytes.setKernelArg(d.m_kernelInitHashTable, 0);
		d.m_memHashTable.setKernelArg(d.m_kernelInitHashTable, 1);
		CLMemory<cl_uchar>::setKernelArg(d.m_kernelInitHashTable, 2, m_mode.extended);
	} else {
		d.m_memPrecomp.setKernelArg(d.m_kernelInitHashTable, 0);
		d.m_memSeed.setKernelArg(d.m_kernelInitHashTable, 1);
		d.m_memHashTable.setKernelArg(d.m_kernelInitHashTable, 2);
		d.m_memPublicAddress.setKernelArg(d.m_kernelInitHashTable, 3);
		CLMemory<cl_uchar>::setKernelArg(d.m_kernelInitHashTable, 4, m_mode.extended);
	}

	// Kernel arguments - profanity_inverse
	d.m_memPointsDeltaX.setKernelArg(d.m_kernelInverse, 0);
	d.m_memInversedNegativeDoubleGy.setKernelArg(d.m_kernelInverse, 1);

	// Kernel arguments - profanity_iterate
	d.m_memPointsDeltaX.setKernelArg(d.m_kernelIterate, 0);
	d.m_memInversedNegativeDoubleGy.setKernelArg(d.m_kernelIterate, 1);
	d.m_memPrevLambda.setKernelArg(d.m_kernelIterate, 2);

	// Kernel arguments - profanity_transform_*
	if(d.m_kernelTransform) {
		d.m_memInversedNegativeDoubleGy.setKernelArg(d.m_kernelTransform, 0);
	}

	// Kernel arguments - profanity_clear_results
	d.m_memResult.setKernelArg(d.m_kernelClearResults, 0);

	// Kernel arguments - profanity_score_*
	d.m_memInversedNegativeDoubleGy.setKernelArg(d.m_kernelScore, 0);
	d.m_memResult.setKernelArg(d.m_kernelScore, 1);
	d.m_memData1.setKernelArg(d.m_kernelScore, 2);
	d.m_memData2.setKernelArg(d.m_kernelScore, 3);
	CLMemory<cl_uchar>::setKernelArg(d.m_kernelScore, 4, d.m_clScoreMax); // Updated in handleResult()

	if (d.m_mode.name == "reverse") {
		d.m_memHashTable.setKernelArg(d.m_kernelScore, 5);
		CLMemory<cl_uchar>::setKernelArg(d.m_kernelScore, 6, m_mode.extended);
	}

	// Seed device
	if (d.m_mode.name == "reverse" || d.m_mode.name == "hashTable") {
		initHashTableContinue(d);
	} else {
		initContinue(d);
	}
}

cl_ulong4 getPrivateKey(size_t seed) {
	std::mt19937_64 eng(seed);
	std::uniform_int_distribution<cl_ulong> distr;

	cl_ulong4 r;
	r.s[0] = distr(eng);
	r.s[1] = distr(eng);
	r.s[2] = distr(eng);
	r.s[3] = distr(eng);
	return r;
}

Dispatcher::Device::Address::Address(): c(0), d(0), e(0) {}

Dispatcher::Device::Address::Address(uint c, uint d, uint e): c(c), d(d), e(e) {}

Dispatcher::Device::Address::Address(uint x[]): c(x[0]), d(x[1]), e(x[2]) {}

bool Dispatcher::Device::Address::operator <(const Address& x) const {
	return (c < x.c) || (c == x.c && d < x.d) || (c == x.c && d == x.d && e < x.e);
}

bool Dispatcher::Device::Address::operator==(const Address &other) const {
	return c == other.c && d == other.d && e == other.e;	
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::size_t Dispatcher::Device::AddressHasher::operator()(const Address& x) const {
	size_t seed = 0;
	hash_combine(seed, x.c);
	hash_combine(seed, x.d);
	hash_combine(seed, x.e);
	return seed;
}

void Dispatcher::initHashTableContinue(Device & d) {
	cl_event event;
	d.m_memPublicAddress.read(false, &event);

	if (d.m_index == 0) {
		const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeInitStart).count();
		const float progress = m_sizeHashTableInitDone * 100.0 / m_sizeHashTableInitTotal;
		const size_t remaining = (100.0 - progress) * (seconds / progress);
		std::cout << "  " << size_t(progress) << "% (remaining " << std::setfill(' ') << std::setw(3) << (progress > 0 ? remaining : 0) << ")" << "\r" << std::flush;
	}

	const size_t sizeLeft = m_HashTableSize - d.m_sizeHashTableInitialized;
	if (sizeLeft && d.m_iterHashTableInitialized > 0) {
		const size_t sizeRun = std::min(d.m_mode.cache ? HASH_TABLE_LOAD_SIZE : HASH_TABLE_JOB_SIZE, std::min(sizeLeft, m_worksizeMax));

		if (d.m_mode.cache) {
			d.m_memPublicBytes.write(true);
		} else {
			d.m_memSeed.write(true);
		}
	
		const auto resEnqueue = clEnqueueNDRangeKernel(d.m_clQueue, d.m_kernelInitHashTable, 1, &d.m_sizeHashTableInitialized, &sizeRun, NULL, 0, NULL, NULL);
		OpenCLException::throwIfError("kernel queueing failed during initilization", resEnqueue);

		std::lock_guard<std::mutex> lock(m_mutex);
		d.m_sizeHashTableInitialized += sizeRun;
		m_sizeHashTableInitDone += sizeRun;
	}
	
	clFlush(d.m_clQueue); 

	const auto resCallback = clSetEventCallback(event, CL_COMPLETE, staticCallback, &d);
	OpenCLException::throwIfError("failed to set custom callback during hash table initialization", resCallback);
}

void Dispatcher::initContinue(Device & d) {
	size_t sizeLeft = m_size - d.m_sizeInitialized;
	const size_t sizeInitLimit = m_size / 20;

	// Print progress
	if (m_mode.name != "reverse" && m_mode.name != "hashTable") {
		const size_t percentDone = m_sizeInitDone * 100 / m_sizeInitTotal;
		std::cout << "  " << percentDone << "%\r" << std::flush;
	}

	if (sizeLeft) {
		cl_event event;
		const size_t sizeRun = std::min(sizeInitLimit, std::min(sizeLeft, m_worksizeMax));
		const auto resEnqueue = clEnqueueNDRangeKernel(d.m_clQueue, d.m_kernelInit, 1, &d.m_sizeInitialized, &sizeRun, NULL, 0, NULL, &event);
		OpenCLException::throwIfError("kernel queueing failed during initilization", resEnqueue);

		// See: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clSetEventCallback.html
		// If an application needs to wait for completion of a routine from the above list in a callback, please use the non-blocking form of the function, and
		// assign a completion callback to it to do the remainder of your work. Note that when a callback (or other code) enqueues commands to a command-queue,
		// the commands are not required to begin execution until the queue is flushed. In standard usage, blocking enqueue calls serve this role by implicitly
		// flushing the queue. Since blocking calls are not permitted in callbacks, those callbacks that enqueue commands on a command queue should either call
		// clFlush on the queue before returning or arrange for clFlush to be called later on another thread.
		clFlush(d.m_clQueue); 

		std::lock_guard<std::mutex> lock(m_mutex);
		d.m_sizeInitialized += sizeRun;
		m_sizeInitDone += sizeRun;

		const auto resCallback = clSetEventCallback(event, CL_COMPLETE, staticCallback, &d);
		OpenCLException::throwIfError("failed to set custom callback during initialization", resCallback);
	} else {
		// Printing one whole string at once helps in avoiding garbled output when executed in parallell
		// const std::string strOutput = "  GPU" + toString(d.m_index) + " initialized";
		// std::cout << strOutput << std::endl;
		clSetUserEventStatus(d.m_eventFinished, CL_COMPLETE);
	}
}

void Dispatcher::enqueueKernel(cl_command_queue & clQueue, cl_kernel & clKernel, size_t worksizeGlobal, const size_t worksizeLocal, cl_event * pEvent = NULL) {
	const size_t worksizeMax = m_worksizeMax;
	size_t worksizeOffset = 0;
	while (worksizeGlobal) {
		const size_t worksizeRun = std::min(worksizeGlobal, worksizeMax);
		const size_t * const pWorksizeLocal = (worksizeLocal == 0 ? NULL : &worksizeLocal);
		const auto res = clEnqueueNDRangeKernel(clQueue, clKernel, 1, &worksizeOffset, &worksizeRun, pWorksizeLocal, 0, NULL, pEvent);
		OpenCLException::throwIfError("kernel queueing failed", res);

		worksizeGlobal -= worksizeRun;
		worksizeOffset += worksizeRun;
	}
}

void Dispatcher::enqueueKernelDevice(Device & d, cl_kernel & clKernel, size_t worksizeGlobal, cl_event * pEvent = NULL) {
	try {
		enqueueKernel(d.m_clQueue, clKernel, worksizeGlobal, d.m_worksizeLocal, pEvent);
	} catch ( OpenCLException & e ) {
		// If local work size is invalid, abandon it and let implementation decide
		if ((e.m_res == CL_INVALID_WORK_GROUP_SIZE || e.m_res == CL_INVALID_WORK_ITEM_SIZE) && d.m_worksizeLocal != 0) {
			std::cout << std::endl << "warning: local work size abandoned on GPU" << d.m_index << std::endl;
			d.m_worksizeLocal = 0;
			enqueueKernel(d.m_clQueue, clKernel, worksizeGlobal, d.m_worksizeLocal, pEvent);
		}
		else {
			throw;
		}
	}
}

void Dispatcher::dispatch(Device & d) {
	cl_event event;
	d.m_memResult.read(false, &event);

	enqueueKernelDevice(d, d.m_kernelInverse, m_size / m_inverseSize);
	enqueueKernelDevice(d, d.m_kernelIterate, m_size);

	if (d.m_kernelTransform) {
		enqueueKernelDevice(d, d.m_kernelTransform, m_size);
	}

	if (d.m_mode.name == "reverse") {
		enqueueKernelDevice(d, d.m_kernelClearResults, d.m_worksizeLocal);
	}

	enqueueKernelDevice(d, d.m_kernelScore, m_size);
	clFlush(d.m_clQueue);

	const auto res = clSetEventCallback(event, CL_COMPLETE, staticCallback, &d);
	OpenCLException::throwIfError("failed to set custom callback", res);
}

void Dispatcher::handleReverse(Device & d) {
	for (auto i = PROFANITY_MAX_SCORE; i > 0; --i) {
		result & r = d.m_memResult[i];
	
		if (r.found > 0) {
			uint a[3];
			const cl_uchar* h = r.foundHash;
			for (size_t i = 0; i < 3; ++i) {
				a[i] = 0;
				a[i] |= ((1U * h[4 * i + 3]) << 24);
				a[i] |= ((1U * h[4 * i + 2]) << 16);
				a[i] |= ((1U * h[4 * i + 1]) << 8);
				a[i] |= ((1U * h[4 * i + 0]) << 0);
			}
			
			Device::Address key(a);
			std::cout << std::endl << "Found candidate!" << std::endl << std::endl;
			
			size_t foundIndex = d.m_addresses.size();
			for (size_t i = 0; i < d.m_addresses.size(); ++i) {
				if (d.m_addresses[i] == key) {
					foundIndex = i;
					break;
				}
			}

			if (foundIndex != d.m_addresses.size()) {
				std::lock_guard<std::mutex> lock(m_mutex);
				
				if (m_clScoreMax != PROFANITY_MAX_SCORE) {
					m_clScoreMax = PROFANITY_MAX_SCORE;
					m_quit = true;
					
					const size_t offset = d.m_batchIndex * m_HashTableSize;
					size_t seed = foundIndex + offset;
					cl_ulong4 rootKey = getPrivateKey(seed);

					// Time delta
					const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart).count();
					// Format private key
					cl_ulong4 privateKey = restorePrivateKey(rootKey, r.foundId, BATCH_X_SIZE * m_batchX + d.m_round - 2);
					const std::string strPrivate = privateKeyToStr(privateKey);
					// Print
					const std::string strVT100ClearLine = "\33[2K\r";
					std::cout << "Id: " << r.foundId << " Round: " << BATCH_X_SIZE * m_batchX + d.m_round - 2 << std::endl;
					std::cout << strVT100ClearLine << "Time: " << std::setw(5) << seconds << " Private: 0x" << strPrivate << std::endl;
				}
			}
		}
	}
}

void Dispatcher::handleResult(Device & d) {
	for (auto i = PROFANITY_MAX_SCORE; i > m_clScoreMax; --i) {
		result & r = d.m_memResult[i];

		if (r.found > 0 && i >= d.m_clScoreMax) {
			d.m_clScoreMax = i;
			CLMemory<cl_uchar>::setKernelArg(d.m_kernelScore, 4, d.m_clScoreMax);

			std::lock_guard<std::mutex> lock(m_mutex);
			if (i >= m_clScoreMax) {
				m_clScoreMax = i;

				if (m_clScoreQuit && i >= m_clScoreQuit) {
					m_quit = true;
				}

				printResult(d.m_clSeed, d.m_round, r, i, timeStart, m_mode);
			}

			break;
		}
	}
}

void Dispatcher::onEvent(cl_event event, cl_int status, Device & d) {
	if (status != CL_COMPLETE) {
		std::cout << "Dispatcher::onEvent - Got bad status: " << status << std::endl;
	}
	else if (d.m_eventFinished != NULL) {
		if (d.m_mode.name == "reverse" || d.m_mode.name == "hashTable") {
			const size_t jobSize = (d.m_mode.cache ? HASH_TABLE_LOAD_SIZE : HASH_TABLE_JOB_SIZE);
			const size_t iterTotal = m_HashTableSize / jobSize;
			const size_t iterDone = d.m_sizeHashTableInitialized / jobSize;		

			if (d.m_mode.cache) {
				if (d.m_iterHashTableInitialized == 0) {
					std::string filename = "cache/" + toString(d.m_batchIndex) + ".bin";
					readAddresses(filename, d.m_addresses);

					// const size_t offset = d.m_batchIndex * m_HashTableSize;
					// for (size_t i = 0; i < d.m_addresses.size(); ++i) {
					// 	d.m_addressToIndex[d.m_addresses[i]] = offset + i;
					// }
				}

				if (iterDone < iterTotal) {
					const size_t offset = iterDone * HASH_TABLE_LOAD_SIZE;
					for (size_t i = 0; i < HASH_TABLE_LOAD_SIZE; ++i) {
						d.m_memPublicBytes[3 * i + 0] = d.m_addresses[offset + i].c;
						d.m_memPublicBytes[3 * i + 1] = d.m_addresses[offset + i].d;
						d.m_memPublicBytes[3 * i + 2] = d.m_addresses[offset + i].e;
					}
				}
			} else {
				if (d.m_iterHashTableInitialized > 1) {
					const size_t localOffset = (d.m_iterHashTableInitialized - 2) * HASH_TABLE_JOB_SIZE;
					const size_t globalOffset = d.m_batchIndex * m_HashTableSize + localOffset;

					for (size_t i = 0; i < HASH_TABLE_JOB_SIZE; ++i) {
						Device::Address key(&d.m_memPublicAddress[3 * i]);
						d.m_addresses[localOffset + i] = key;
						// d.m_addressToIndex[key] = globalOffset + i;
					}
				}

				if (iterDone < iterTotal) {
					const size_t offset = d.m_batchIndex * m_HashTableSize + HASH_TABLE_JOB_SIZE * iterDone;
					for (size_t i = 0; i < HASH_TABLE_JOB_SIZE; ++i) {
						d.m_memSeed[i] = getPrivateKey(offset + i);
					}
				}
			}

			if (d.m_iterHashTableInitialized <= iterTotal) {
				++d.m_iterHashTableInitialized;
				initHashTableContinue(d);
			} else {
				if (m_mode.name == "hashTable") {
					std::string filename = "cache/" + toString(d.m_batchIndex) + ".bin";
					writeAddresses(filename, d.m_addresses);
					clSetUserEventStatus(d.m_eventFinished, CL_COMPLETE);
				} else {
					initContinue(d);
				}
			}
		} else {
			initContinue(d);
		}
	} else {
		++d.m_round;
		
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			m_stepsDone += 1;

			d.m_speed.sample(m_size);
			if (d.m_index == 0 && !m_quit) {
				printSpeed();
			}
		}

		if (m_mode.name == "reverse") {
			handleReverse(d);
		} else {
			handleResult(d);
		}

		bool bDispatch = true;
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			if (m_quit || d.m_round == BATCH_X_SIZE + STEPS_OFFSET) {
				bDispatch = false;
				if(--m_countRunning == 0) {
					clSetUserEventStatus(m_eventFinished, CL_COMPLETE);
				}
			}
		}

		if (bDispatch) {
			dispatch(d);
		}
	}
}

// This is run when m_mutex is held.
void Dispatcher::printSpeed() {
	// ++m_countPrint;
	// if( m_countPrint > m_vDevices.size() ) {
	std::string strGPUs;
	double speedTotal = 0;
	unsigned int i = 0;
	for (auto & e : m_vDevices) {
		const auto curSpeed = e->m_speed.getSpeed();
		speedTotal += curSpeed;
		strGPUs += " GPU" + toString(e->m_index) + ": " + formatSpeed(curSpeed);
		++i;
	}

	std::string strProgress;
	std::string strTime;
	if (m_mode.name == "reverse") {
		const float progress = 100.0 * m_stepsDone / m_stepsTotal;
		const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeRunStart).count();
		const size_t remaining = (100.0 - progress) * (seconds / progress);
		
		std::ostringstream strProgressBuilder;
		strProgressBuilder << "Batch x=" << m_batchX + 1 << " y=" << m_batchY + 1 << ": " << std::setfill(' ') << std::setw(3) << size_t(progress) << "%:";
		strProgress = strProgressBuilder.str();

		std::ostringstream strTimeBuilder;
		strTimeBuilder << " (remaining " << std::setfill(' ') << std::setw(3) << remaining << ")";
		strTime = strTimeBuilder.str();
	}

	const std::string strVT100ClearLine = "\33[2K\r";
	std::cerr << strVT100ClearLine << strProgress << " total: " << formatSpeed(speedTotal) << " -" << strGPUs << strTime << '\r' << std::flush;
	// m_countPrint = 0;
	// }
}

void CL_CALLBACK Dispatcher::staticCallback(cl_event event, cl_int event_command_exec_status, void * user_data) {
	Device * const pDevice = static_cast<Device *>(user_data);
	pDevice->m_parent.onEvent(event, event_command_exec_status, *pDevice);
	clReleaseEvent(event);
}

std::string Dispatcher::formatSpeed(double f) {
	const std::string S = " KMGT";

	unsigned int index = 0;
	while (f > 1000.0f && index < S.size()) {
		f /= 1000.0f;
		++index;
	}

	std::ostringstream ss;
	ss << std::fixed << std::setprecision(3) << (double)f << " " << S[index] << "H/s";
	return ss.str();
}

void Dispatcher::writeAddresses(std::string& filename, std::vector<Device::Address>& addresses) {
	std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::out);

	for (auto const& address : addresses) {
		file.write((char*)(&address.c), sizeof(uint));
		file.write((char*)(&address.d), sizeof(uint));
		file.write((char*)(&address.e), sizeof(uint));
	}

	file.close();
}

void Dispatcher::readAddresses(std::string& filename, std::vector<Device::Address>& addresses) {
	std::ifstream file;
    file.open(filename, std::ios::binary | std::ios::in);
	
	Device::Address key;
	addresses.resize(m_HashTableSize);
	for (size_t i = 0; i < m_HashTableSize; ++i) {
		file.read((char*)(&key.c), sizeof(uint));
		file.read((char*)(&key.d), sizeof(uint));
		file.read((char*)(&key.e), sizeof(uint));
		addresses[i] = key;
	}

	file.close();
}
