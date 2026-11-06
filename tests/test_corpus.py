"""Checks for the generated training corpus and its C++ examples."""
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import corpus_gen as corpus
from tools.strip_cpp_comments import prepare_source, strip_comments


class CorpusTests(unittest.TestCase):
    def test_full_corpus_training_constraints(self):
        random.seed(1337)
        text = corpus.generate_corpus()
        self.assertEqual(len(set(text)), 95)
        self.assertFalse(set(text) - corpus.ALLOWED_SET)
        anchor = text.find("\n    return 0;\n}\n\n#pragma once\n\n#include <")
        self.assertGreaterEqual(anchor, 0)
        self.assertTrue(text[anchor:anchor + 32].endswith("\n\n"))
        random.seed(1337)
        self.assertEqual(text, corpus.generate_corpus())
        entries = json.loads(corpus.SOURCE_MANIFEST.read_text())
        self.assertEqual({e["project"] for e in entries},
                         {"chromium", "firefox", "thunderbird", "gcc", "llvm"})
        self.assertEqual(len(entries), 41)
        for entry in entries:
            snippet = (corpus.ROOT / entry["snippet"]).read_text(encoding="utf-8")
            self.assertEqual(hashlib.sha256(snippet.encode()).hexdigest(),
                             entry["snippet_sha256"])
            self.assertEqual(strip_comments(snippet), snippet, entry["snippet"])
            self.assertEqual(text.count(snippet), 1, entry["snippet"])
            self.assertIn(entry["commit"], entry["source_url"])
        for name in ("FixedStack", "Counter", "ByteView", "is_power_of_two",
                     "find_byte", "trim_spaces", "saturating_add"):
            self.assertIn(name, text)

    def test_comment_stripping_preserves_literals(self):
        code = ("auto url = \"https://example.test/a/*b*/\"; // comment\n"
                "auto raw = R\"tag(// not a comment /* either */)tag\";\n"
                "int n = 1'000; /* block\ncomment */ int m = 2;\n"
                "char slash = '/'; // continued \\\ncomment\n")
        result = strip_comments(code)
        self.assertIn('\"https://example.test/a/*b*/\"', result)
        self.assertIn('R\"tag(// not a comment /* either */)tag\"', result)
        self.assertIn("1'000", result)
        self.assertNotIn('block', result)
        self.assertNotIn('continued', result)
        self.assertEqual(code.count('\n'), result.count('\n'))
        self.assertEqual(prepare_source('int a;\n\n\n\nint b;\n'),
                         'int a;\n\n\nint b;\n')
        with self.assertRaises(ValueError):
            prepare_source('auto s = R\"(first\nsecond)\";\n')

    @unittest.skipUnless(shutil.which("g++"), "g++ is required")
    def test_sample_translation_units_compile(self):
        random.seed(1337)
        for i in range(20):
            # Compile header-style units as source, without pragma-once warnings.
            text = corpus.gen_translation_unit().replace("#pragma once\n", "")
            result = subprocess.run(
                ["g++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                 "-pedantic", "-x", "c++", "-fsyntax-only", "-"],
                input=text, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, f"unit {i}: {result.stderr}")

    @unittest.skipUnless(shutil.which("g++"), "g++ is required")
    def test_new_examples_compile_and_run(self):
        builders = [corpus.gen_class_fixed_stack, corpus.gen_class_counter,
                    corpus.gen_class_byte_view, corpus.gen_free_fn_power_of_two,
                    corpus.gen_free_fn_find_byte, corpus.gen_free_fn_trim,
                    corpus.gen_free_fn_saturating_add]
        source = corpus.gen_includes() + "\n#include <cassert>\n\n"
        source += "\n\n".join(builder() for builder in builders)
        source += r'''
int main() {
    FixedStack<int, 2> stack;
    assert(!stack.pop());
    assert(stack.push(4) && stack.push(7));
    assert(!stack.push(8));
    assert(stack.pop() == 7 && stack.pop() == 4 && !stack.pop());
    Counter counter;
    counter.increment();
    assert(counter.load() == 1);
    counter.reset();
    assert(counter.load() == 0);
    const uint8_t data[] = {3, 5, 7};
    ByteView view{std::span<const uint8_t>(data)};
    assert(view.size() == 3 && view.get(1) == 5 && !view.get(3));
    assert(find_byte(data, 5) == 1 && !find_byte(data, 9));
    assert(trim_spaces("  hi  ") == "hi");
    assert(trim_spaces("   ").empty());
    static_assert(is_power_of_two(8U) && !is_power_of_two(0U));
    static_assert(!is_power_of_two(7U));
    static_assert(saturating_add<uint8_t>(250, 10) == 255);
    static_assert(saturating_add<uint8_t>(2, 3) == 5);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            executable = str(Path(tmp) / "examples")
            result = subprocess.run(
                ["g++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                 "-pedantic", "-x", "c++", "-o", executable, "-"],
                input=source, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([executable], check=True)


if __name__ == "__main__":
    unittest.main()
