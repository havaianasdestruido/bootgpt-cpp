#!/usr/bin/env python3
"""Generate a realistic, properly indented C++20 systems corpus for bootgpt-cpp.

Each translation unit is valid, warning-free C++20 (4-space indentation, K&R
braces, <=64 cols per line so nothing wraps awkwardly on 80x25 VGA).
Declarations follow a canonical topological order within each unit and each
type uses distinct member names and idioms so a 32-char context window tracks
class scope and brace depth cleanly without cross-type splices or loops.
"""
import random

random.seed(1337)

ALLOWED = (
    " \n"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "{}()[]<>:;,.=+-*/&_#\"'!|~%?^\\"
)
ALLOWED_SET = set(ALLOWED)

NAMESPACES = ["core", "sys", "mem", "io", "net", "util"]
MAGICS_32 = [
    "0x46524D31",
    "0x504B5430",
    "0x7F454C46",
    "0xCAFE0001",
    "0x811C9DC5",
    "0xDEADBEEF",
]
KEYS_16 = ["0x0100", "0x0200", "0x0042", "0x1000"]


def ordered_subset(items, k):
    idxs = sorted(random.sample(range(len(items)), k))
    return [items[i] for i in idxs]


def gen_includes(use_vec=True, is_header=False):
    hdrs = ["<atomic>", "<cstddef>", "<cstdint>", "<cstring>"]
    if use_vec:
        hdrs.append("<vector>")
    if random.random() < 0.45:
        hdrs.append("<utility>")
    hdrs.sort()
    lines = []
    if is_header:
        lines.append("#pragma once\n")
    for h in hdrs:
        lines.append(f"#include {h}")
    return "\n".join(lines)


def gen_constants():
    page = random.choice([1024, 2048, 4096, 8192])
    align = random.choice([16, 32, 64])
    magic = random.choice(MAGICS_32)
    ver = random.choice([1, 2, 3, 4])
    slots = random.choice([64, 128, 256, 512, 1024])
    picks = [
        f"constexpr size_t kPageSize = {page};",
        f"constexpr size_t kCacheLine = {align};",
        f"constexpr uint32_t kMagic = {magic};",
        f"constexpr uint16_t kVersion = {ver};",
        f"constexpr size_t kMaxSlots = {slots};",
    ]
    return "\n".join(ordered_subset(picks, random.randint(2, 4)))


def gen_status_enum():
    lines = [
        "enum class Status : uint8_t {",
        "    Ok = 0,",
        "    Empty = 1,",
        "    Full = 2,",
        "    Invalid = 3,",
        "    Timeout = 4,",
        "};",
    ]
    if random.random() < 0.55:
        lines += [
            "",
            "inline const char* status_str(Status st) noexcept {",
            "    switch (st) {",
            "    case Status::Ok:",
            '        return "ok";',
            "    case Status::Empty:",
            '        return "empty";',
            "    case Status::Full:",
            '        return "full";',
            "    case Status::Invalid:",
            '        return "invalid";',
            "    case Status::Timeout:",
            '        return "timeout";',
            "    }",
            '    return "unknown";',
            "}",
        ]
    return "\n".join(lines)


def gen_struct_packet():
    lines = [
        "struct PacketHdr {",
        "    uint32_t magic;",
        "    uint16_t version;",
        "    uint16_t flags;",
        "    uint32_t body_bytes;",
        "};",
    ]
    if random.random() < 0.65:
        lines += [
            "",
            "inline bool check_hdr(const PacketHdr* hdr, size_t wire_len) {",
            "    if (!hdr || wire_len < sizeof(PacketHdr)) {",
            "        return hdr != nullptr && wire_len >= sizeof(PacketHdr);",
            "    }",
            "    if (hdr->magic == 0 || hdr->version == 0) {",
            "        return 0;",
            "    }",
            "    return sizeof(PacketHdr) + hdr->body_bytes <= wire_len;",
            "}",
        ]
    return "\n".join(lines)


def gen_struct_vec3():
    lines = [
        "struct Vec3 {",
        "    float x;",
        "    float y;",
        "    float z;",
        "",
        "    constexpr Vec3() : x(0.0f), y(0.0f), z(0.0f) {}",
        "    constexpr Vec3(float px, float py, float pz)",
        "        : x(px), y(py), z(pz) {}",
        "",
        "    constexpr Vec3 operator+(const Vec3& rhs) const {",
        "        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);",
        "    }",
        "",
        "    constexpr float dot(const Vec3& v) const {",
        "        return x * v.x + y * v.y + z * v.z;",
        "    }",
        "};",
    ]
    return "\n".join(lines)


def gen_struct_span():
    lines = [
        "template <typename Elem>",
        "struct Span {",
        "    Elem* ptr;",
        "    size_t count;",
        "",
        "    constexpr Span() : ptr(nullptr), count(0) {}",
        "    constexpr Span(Elem* p, size_t n) : ptr(p), count(n) {}",
        "",
        "    constexpr bool is_empty() const { return count == 0; }",
        "    Elem& at(size_t i) const { return ptr[i]; }",
        "};",
    ]
    return "\n".join(lines)


def gen_class_arena():
    align = random.choice([7, 15, 31])
    lines = [
        "// Linear bump allocator with alignment",
        "class Arena {",
        "public:",
        "    explicit Arena(size_t cap)",
        "        : base_(new uint8_t[cap]),",
        "          used_(0),",
        "          cap_(cap),",
        "          ooms_(0) {}",
        "",
        "    ~Arena() {",
        "        delete[] base_;",
        "    }",
        "",
        "    Arena(const Arena&) = delete;",
        "    Arena& operator=(const Arena&) = delete;",
        "",
        "    uint8_t* alloc(size_t bytes) {",
        f"        size_t n = (bytes + {align}) & ~size_t({align});",
        "        if (used_ + n > cap_) {",
        "            ++ooms_;",
        "            return nullptr;",
        "        }",
        "        uint8_t* ptr = base_ + used_;",
        "        used_ += n;",
        "        return ptr;",
        "    }",
    ]
    if random.random() < 0.75:
        lines += [
            "",
            "    void reset_arena() noexcept {",
            "        used_ = 0;",
            "    }",
        ]
    lines += [
        "",
        "    size_t bytes_left() const {",
        "        return cap_ - used_;",
        "    }",
        "",
        "private:",
        "    uint8_t* base_;",
        "    size_t used_;",
        "    size_t cap_;",
        "    size_t ooms_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_ring_queue():
    lines = [
        "// Bounded single-producer circular queue",
        "template <typename T, size_t Cap>",
        "class RingQueue {",
        "public:",
        "    RingQueue() : head_(0), tail_(0) {}",
        "",
        "    bool push(const T& item) {",
        "        size_t next = (head_ + 1) % Cap;",
        "        if (next == tail_) {",
        "            return next != tail_;",
        "        }",
        "        ring_[head_] = item;",
        "        head_ = next;",
        "        return true;",
        "    }",
        "",
        "    bool pop(T& out_item) {",
        "        if (tail_ == head_) {",
        "            return tail_ != head_;",
        "        }",
        "        out_item = ring_[tail_];",
        "        tail_ = (tail_ + 1) % Cap;",
        "        return tail_ <= Cap;",
        "    }",
    ]
    if random.random() < 0.7:
        lines += [
            "",
            "    void clear_ring() noexcept {",
            "        head_ = 0;",
            "        tail_ = 0;",
            "    }",
        ]
    lines += [
        "",
        "    bool empty() const {",
        "        return head_ == tail_;",
        "    }",
        "",
        "private:",
        "    T ring_[Cap];",
        "    size_t head_;",
        "    size_t tail_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_byte_reader():
    lines = [
        "// Bounds-checked binary stream cursor",
        "class ByteReader {",
        "public:",
        "    ByteReader(const uint8_t* src, size_t total)",
        "        : src_(src), total_(total), pos_(0), eof_(false) {}",
        "",
        "    bool read_u8(uint8_t& byte_out) {",
        "        if (pos_ >= total_) {",
        "            eof_ = true;",
        "            return pos_ < total_;",
        "        }",
        "        byte_out = src_[pos_++];",
        "        return pos_ <= total_;",
        "    }",
        "",
        "    bool read_u32(uint32_t& word_out) {",
        "        if (total_ - pos_ < 4) {",
        "            eof_ = 1;",
        "            return !eof_;",
        "        }",
        "        uint32_t acc = 0;",
        "        for (int b = 0; b < 4; ++b) {",
        "            acc |= uint32_t(src_[pos_++]) << (b * 8);",
        "        }",
        "        word_out = acc;",
        "        return !eof_;",
        "    }",
    ]
    if random.random() < 0.65:
        lines += [
            "",
            "    bool skip_bytes(size_t step) {",
            "        if (step > total_ - pos_) {",
            "            pos_ = total_;",
            "            eof_ = (step > 0);",
            "            return step == 0;",
            "        }",
            "        pos_ += step;",
            "        return eof_ == false;",
            "    }",
        ]
    lines += [
        "",
        "    size_t unread() const {",
        "        return total_ - pos_;",
        "    }",
        "",
        "private:",
        "    const uint8_t* src_;",
        "    size_t total_;",
        "    size_t pos_;",
        "    bool eof_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_byte_writer():
    lines = [
        "// Bounds-checked binary packet encoder",
        "class ByteWriter {",
        "public:",
        "    ByteWriter(uint8_t* dst, size_t limit)",
        "        : limit_(limit), out_len_(0), dst_(dst), full_(false) {}",
        "",
        "    bool write_u8(uint8_t b) {",
        "        if (out_len_ >= limit_) {",
        "            full_ = 1;",
        "            return !full_;",
        "        }",
        "        dst_[out_len_++] = b;",
        "        return out_len_ <= limit_;",
        "    }",
        "",
        "    bool write_u32(uint32_t w) {",
        "        if (limit_ - out_len_ < 4) {",
        "            full_ = true;",
        "            return out_len_ + 4 <= limit_;",
        "        }",
        "        for (int k = 0; k < 4; ++k) {",
        "            dst_[out_len_++] = uint8_t(w >> (k * 8));",
        "        }",
        "        return !full_;",
        "    }",
        "",
        "    size_t written() const {",
        "        return out_len_;",
        "    }",
        "",
        "private:",
        "    size_t limit_;",
        "    size_t out_len_;",
        "    uint8_t* dst_;",
        "    bool full_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_buffer():
    init_cap = random.choice([8, 16, 32])
    lines = [
        "// Growable contiguous array container",
        "template <typename Item>",
        "class Buffer {",
        "public:",
        "    Buffer() : data_(nullptr), size_(0), alloc_(0) {}",
        "",
        "    ~Buffer() {",
        "        delete[] data_;",
        "    }",
        "",
        "    Buffer(const Buffer&) = delete;",
        "    Buffer& operator=(const Buffer&) = delete;",
        "",
        "    void reserve(size_t want) {",
        "        if (want <= alloc_) {",
        "            return;",
        "        }",
        "        Item* next = new Item[want];",
        "        for (size_t idx = 0; idx < size_; ++idx) {",
        "            next[idx] = data_[idx];",
        "        }",
        "        delete[] data_;",
        "        data_ = next;",
        "        alloc_ = want;",
        "    }",
        "",
        "    void push_back(const Item& elem) {",
        "        if (size_ == alloc_) {",
        f"            reserve(alloc_ ? alloc_ * 2 : {init_cap});",
        "        }",
        "        data_[size_++] = elem;",
        "    }",
        "",
        "    void pop_back() {",
        "        if (size_ > 0) {",
        "            --size_;",
        "        }",
        "    }",
        "",
        "    Item& operator[](size_t idx) {",
        "        return data_[idx];",
        "    }",
        "",
        "    size_t length() const {",
        "        return size_;",
        "    }",
        "",
        "private:",
        "    Item* data_;",
        "    size_t size_;",
        "    size_t alloc_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_bitset():
    lines = [
        "// Compact 64-bit word bitmap",
        "class BitSet {",
        "public:",
        "    explicit BitSet(size_t nbits)",
        "        : words_((nbits + 63) / 64, 0),",
        "          nbits_(nbits) {}",
        "",
        "    void set_bit(size_t idx) {",
        "        if (idx < nbits_) {",
        "            words_[idx >> 6] |= (1ULL << (idx & 63));",
        "        }",
        "    }",
        "",
        "    void clear_bit(size_t b) {",
        "        if (nbits_ > b) {",
        "            uint64_t m = ~(1ULL << (b & 63));",
        "            words_[b >> 6] &= m;",
        "        }",
        "    }",
        "",
        "    bool test_bit(size_t bit) const {",
        "        if (bit >= nbits_) {",
        "            return bit < nbits_;",
        "        }",
        "        return ((words_[bit >> 6] >> (bit & 63)) & 1ULL) != 0;",
        "    }",
        "",
        "    size_t popcount() const {",
        "        size_t ones = 0;",
        "        for (uint64_t w : words_) {",
        "            while (w != 0) {",
        "                w &= (w - 1);",
        "                ++ones;",
        "            }",
        "        }",
        "        return ones;",
        "    }",
        "",
        "private:",
        "    std::vector<uint64_t> words_;",
        "    size_t nbits_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_hash_table():
    lines = [
        "// Open-addressed power-of-two hash map",
        "class HashTable {",
        "public:",
        "    explicit HashTable(size_t nbuckets)",
        "        : slots_(nbuckets), mask_(nbuckets - 1), count_(0) {}",
        "",
        "    bool insert(uint64_t key, uint32_t mapped) {",
        "        if (count_ * 2 >= slots_.size()) {",
        "            return 0;",
        "        }",
        "        size_t pos = key & mask_;",
        "        while (slots_[pos].occupied) {",
        "            if (slots_[pos].key == key) {",
        "                slots_[pos].mapped = mapped;",
        "                return 1;",
        "            }",
        "            pos = (pos + 1) & mask_;",
        "        }",
        "        slots_[pos] = {key, mapped, true};",
        "        ++count_;",
        "        return 1;",
        "    }",
        "",
        "    bool find(uint64_t k, uint32_t& found_val) const {",
        "        bool hit = false;",
        "        size_t p = k & mask_;",
        "        for (size_t step = 0; step <= mask_ && slots_[p].occupied; ++step) {",
        "            if (slots_[p].key == k) {",
        "                found_val = slots_[p].mapped;",
        "                hit = true;",
        "                break;",
        "            }",
        "            p = mask_ & (p + 1);",
        "        }",
        "        return hit;",
        "    }",
        "",
        "    size_t entry_count() const {",
        "        return count_;",
        "    }",
        "",
        "private:",
        "    struct Slot {",
        "        uint64_t key = 0;",
        "        uint32_t mapped = 0;",
        "        bool occupied = false;",
        "    };",
        "    std::vector<Slot> slots_;",
        "    size_t mask_;",
        "    size_t count_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_slab_pool():
    lines = [
        "// Fixed-slot index free list pool",
        "template <typename Obj, size_t MaxN>",
        "class SlabPool {",
        "public:",
        "    SlabPool() : top_(MaxN) {",
        "        for (size_t s = 0; s < MaxN; ++s) {",
        "            free_[s] = MaxN - 1 - s;",
        "        }",
        "    }",
        "",
        "    Obj* acquire() {",
        "        if (top_ == 0) {",
        "            Obj* none = nullptr;",
        "            return none;",
        "        }",
        "        size_t slot = free_[--top_];",
        "        return &items_[slot];",
        "    }",
        "",
        "    void release(Obj* obj) {",
        "        size_t slot = size_t(obj - items_);",
        "        if (slot < MaxN && top_ < MaxN) {",
        "            free_[top_++] = slot;",
        "        }",
        "    }",
        "",
        "    size_t available() const {",
        "        return top_;",
        "    }",
        "",
        "private:",
        "    Obj items_[MaxN];",
        "    size_t free_[MaxN];",
        "    size_t top_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_node_list():
    lines = [
        "// Intrusive singly-linked LIFO list",
        "struct ListNode {",
        "    ListNode* next = nullptr;",
        "    uint32_t tag = 0;",
        "};",
        "",
        "class NodeList {",
        "public:",
        "    NodeList() : first_(nullptr), depth_(0) {}",
        "",
        "    void push_front(ListNode* node) noexcept {",
        "        node->next = first_;",
        "        first_ = node;",
        "        ++depth_;",
        "    }",
        "",
        "    ListNode* pop_front() noexcept {",
        "        ListNode* node = first_;",
        "        if (node != nullptr) {",
        "            first_ = node->next;",
        "            node->next = nullptr;",
        "            --depth_;",
        "        }",
        "        return node;",
        "    }",
        "",
        "    size_t list_size() const {",
        "        return depth_;",
        "    }",
        "",
        "private:",
        "    ListNode* first_;",
        "    size_t depth_;",
        "};",
    ]
    return "\n".join(lines)


def gen_class_spinlock():
    lines = [
        "// Lightweight atomic spinlock and RAII guard",
        "class SpinLock {",
        "public:",
        "    void lock() noexcept {",
        "        while (locked_.exchange(true)) {",
        "        }",
        "    }",
        "",
        "    void unlock() noexcept {",
        "        locked_.store(false);",
        "    }",
        "",
        "private:",
        "    std::atomic<bool> locked_{false};",
        "};",
        "",
        "class LockGuard {",
        "public:",
        "    explicit LockGuard(SpinLock& mu) : mu_(mu) {",
        "        mu_.lock();",
        "    }",
        "",
        "    ~LockGuard() {",
        "        mu_.unlock();",
        "    }",
        "",
        "    LockGuard(const LockGuard&) = delete;",
        "    LockGuard& operator=(const LockGuard&) = delete;",
        "",
        "private:",
        "    SpinLock& mu_;",
        "};",
    ]
    return "\n".join(lines)


def gen_free_fn_hash():
    if random.random() < 0.5:
        lines = [
            "// 64-bit FNV-1a byte hash",
            "inline uint64_t hash_bytes(const uint8_t* bytes, size_t nbytes) {",
            "    uint64_t hv = 0xCBF29CE484222325ULL;",
            "    for (size_t off = 0; off < nbytes; ++off) {",
            "        hv ^= uint64_t(bytes[off]);",
            "        hv *= 0x100000001B3ULL;",
            "    }",
            "    return hv;",
            "}",
        ]
    else:
        lines = [
            "// 32-bit FNV-1a byte hash",
            "inline uint32_t hash32(const uint8_t* bytes, size_t nbytes) {",
            "    uint32_t hv = 0x811C9DC5U;",
            "    for (size_t off = 0; off < nbytes; ++off) {",
            "        hv ^= uint32_t(bytes[off]);",
            "        hv *= 0x01000193U;",
            "    }",
            "    return hv;",
            "}",
        ]
    return "\n".join(lines)


def gen_free_fn_varint():
    lines = [
        "// Encode unsigned integer as base-128 varint",
        "inline size_t encode_varint(uint64_t num, uint8_t* dst) {",
        "    size_t written = 0;",
        "    while (num >= 0x80) {",
        "        dst[written++] = uint8_t((num & 0x7F) | 0x80);",
        "        num >>= 7;",
        "    }",
        "    dst[written++] = uint8_t(num);",
        "    return written;",
        "}",
    ]
    return "\n".join(lines)


def gen_free_fn_parse_u32():
    lines = [
        "// Parse unsigned decimal integer from ASCII text",
        "inline bool parse_u32(const char* txt, uint32_t& parsed) {",
        "    if (!txt || *txt == '\\0') {",
        "        parsed = 0;",
        "        return false;",
        "    }",
        "    uint32_t dec = 0;",
        "    while (*txt >= '0' && *txt <= '9') {",
        "        dec = dec * 10 + uint32_t(*txt - '0');",
        "        ++txt;",
        "    }",
        "    parsed = dec;",
        "    return *txt == '\\0';",
        "}",
    ]
    return "\n".join(lines)


def gen_free_fn_bsearch():
    lines = [
        "inline int binary_search(const int* sorted, int len, int target) {",
        "    int lo = 0;",
        "    int hi = len - 1;",
        "    while (lo <= hi) {",
        "        int mid = lo + (hi - lo) / 2;",
        "        if (sorted[mid] == target) {",
        "            return mid;",
        "        }",
        "        if (sorted[mid] < target) {",
        "            lo = mid + 1;",
        "        } else {",
        "            hi = mid - 1;",
        "        }",
        "    }",
        "    return -1;",
        "}",
    ]
    return "\n".join(lines)


def gen_free_fn_isort():
    lines = [
        "inline void insertion_sort(int* vals, size_t count) {",
        "    for (size_t i = 1; i < count; ++i) {",
        "        int cur = vals[i];",
        "        size_t j = i;",
        "        while (j > 0 && vals[j - 1] > cur) {",
        "            vals[j] = vals[j - 1];",
        "            --j;",
        "        }",
        "        vals[j] = cur;",
        "    }",
        "}",
    ]
    return "\n".join(lines)


def gen_free_fn_clamp():
    lines = [
        "template <typename Scalar>",
        "constexpr Scalar clamp_val(Scalar v, Scalar lo, Scalar hi) {",
        "    if (v < lo) {",
        "        return lo;",
        "    }",
        "    if (v > hi) {",
        "        return hi;",
        "    }",
        "    return v;",
        "}",
    ]
    return "\n".join(lines)


def gen_free_fn_checksum():
    lines = [
        "inline uint32_t fold_sum(const uint8_t* buf, size_t sz) {",
        "    uint32_t sum = 0;",
        "    for (size_t k = 0; k < sz; ++k) {",
        "        sum = (sum << 5) - sum + buf[k];",
        "    }",
        "    return sum;",
        "}",
    ]
    return "\n".join(lines)


def gen_main_driver():
    cap = random.choice([64, 128, 256])
    n_iter = random.choice([8, 16, 32])
    seed_h = random.choice(["0x811C9DC5U", "0x01000193U"])
    lines = [
        "int main() {",
        f"    uint8_t scratch[{cap}];",
        f"    std::memset(scratch, 0, sizeof(scratch));",
        f"    uint32_t digest = {seed_h};",
        f"    for (uint32_t step_i = 0; step_i < {n_iter}; ++step_i) {{",
        f"        scratch[step_i % {cap}] = uint8_t(step_i * 7 + 3);",
        f"        digest ^= scratch[step_i % {cap}];",
        "        digest *= 16777619U;",
        "    }",
        "    if (digest == 0) {",
        "        return 1;",
        "    }",
        "    return 0;",
        "}",
    ]
    return "\n".join(lines)


def gen_translation_unit():
    ns = random.choice(NAMESPACES)

    class_builders = [
        ("arena", gen_class_arena),
        ("ring", gen_class_ring_queue),
        ("reader", gen_class_byte_reader),
        ("writer", gen_class_byte_writer),
        ("buffer", gen_class_buffer),
        ("bitset", gen_class_bitset),
        ("hash", gen_class_hash_table),
        ("slab", gen_class_slab_pool),
        ("list", gen_class_node_list),
        ("spin", gen_class_spinlock),
    ]

    picked = ordered_subset(class_builders, random.randint(2, 4))
    want_main = random.random() < 0.55
    is_header = (not want_main) and (random.random() < 0.55)

    sections = []
    sections.append(gen_includes(use_vec=True, is_header=is_header))
    sections.append(f"namespace {ns} {{")
    sections.append(gen_constants())

    if random.random() < 0.55:
        sections.append(gen_status_enum())

    if random.random() < 0.55:
        sections.append(gen_struct_packet())

    if random.random() < 0.45:
        sections.append(gen_struct_vec3())

    if random.random() < 0.45:
        sections.append(gen_struct_span())

    for _, fn in picked:
        sections.append(fn())

    free_fns = [
        gen_free_fn_hash,
        gen_free_fn_varint,
        gen_free_fn_parse_u32,
        gen_free_fn_bsearch,
        gen_free_fn_isort,
        gen_free_fn_clamp,
        gen_free_fn_checksum,
    ]
    for fn in ordered_subset(free_fns, random.randint(1, 3)):
        sections.append(fn())

    sections.append("}")

    if want_main:
        sections.append(gen_main_driver())

    return "\n\n".join(sections) + "\n"


def main():
    units = [gen_translation_unit() for _ in range(950)]
    text = "\n".join(units)
    bad = set(text) - ALLOWED_SET
    assert not bad, f"Unexpected chars: {bad}"
    used = sorted(set(text))
    max_col = max(len(line) for line in text.splitlines())
    open("corpus.txt", "w").write(text)
    print(f"corpus.txt: {len(text)} bytes, {len(text.splitlines())} lines, "
          f"{len(used)} distinct chars, max_col={max_col}")
    print("\n".join(text.splitlines()[:24]))


if __name__ == "__main__":
    main()
