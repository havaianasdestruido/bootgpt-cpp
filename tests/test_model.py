"""Check that the bundled model, metadata and boot image agree."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model"


class ModelArtifactTests(unittest.TestCase):
    def test_artifact_layout(self):
        params = json.loads((MODEL / "params.json").read_text())
        inc = dict(re.findall(r"^(\w+)\s+equ\s+([^\s]+)",
                              (MODEL / "params.inc").read_text(), re.M))
        inc = {k: int(v, 0) for k, v in inc.items()}
        vocab, ctx, emb, hidden = (params[k] for k in ("V", "C", "E", "H"))
        for name, value in (("VOCAB", vocab), ("CTX", ctx), ("EMBDIM", emb),
                            ("HID", hidden), ("SECTORS", params["SECTORS"])):
            self.assertEqual(inc[name], value)
        for name in ("A1SHIFT", "A2SHIFT"):
            self.assertEqual(inc[name], params[name])
        ee, ew1, ew2 = params["scales"]
        self.assertEqual(params["A1SHIFT"], ee + ew1 - 12)
        self.assertEqual(params["A2SHIFT"], 6 + ew2 - 8)
        self.assertEqual(inc["W1_OFF"], vocab * (1 + emb))
        self.assertEqual(inc["W2_OFF"], inc["W1_OFF"] + hidden * ctx * emb)
        self.assertEqual(inc["B1_OFF"], inc["W2_OFF"] + vocab * hidden)
        self.assertEqual(inc["B2_OFF"], inc["B1_OFF"] + 2 * hidden)
        self.assertEqual(inc["SEEDADDR"], 0x8000 + inc["B2_OFF"] + 2 * vocab)
        self.assertEqual(params["SAMPLER"], "epsilon-soft-q14")
        weights = (MODEL / "weights.bin").read_bytes()
        self.assertEqual(len(weights), params["SECTORS"] * 512)
        self.assertEqual(len(set(weights[:vocab])), vocab)
        seed_offset = inc["SEEDADDR"] - 0x8000
        self.assertEqual(weights[seed_offset:seed_offset + ctx],
                         bytes(params["seed_ctx"]))
        boot = (MODEL / "boot.bin").read_bytes()
        self.assertEqual(len(boot), 512)
        self.assertEqual(boot[-2:], b"\x55\xaa")
        self.assertEqual((MODEL / "disk.img").read_bytes(), boot + weights)
        with zipfile.ZipFile(MODEL / "model.npz") as model:
            self.assertEqual(set(model.namelist()),
                             {"emb.npy", "W1.npy", "b1.npy", "W2.npy",
                              "b2.npy", "vocab.npy"})
            self.assertIsNone(model.testzip())

    @unittest.skipUnless(shutil.which("nasm"), "nasm is required")
    def test_bundled_highlighter_rebuilds(self):
        inc = (MODEL / "params.inc").read_text()
        offset = int(re.search(r"^SYNTAXADDR\s+equ\s+(\S+)", inc, re.M).group(1), 0) - 0x8000
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "syntax.bin"
            subprocess.run(["nasm", "-f", "bin", "syntax.asm", "-o", str(output)],
                           cwd=ROOT, check=True)
            tail = (MODEL / "weights.bin").read_bytes()[offset:]
            code = output.read_bytes()
            self.assertEqual(tail[:len(code)], code)
            self.assertEqual(tail[len(code):], bytes(len(tail) - len(code)))

    @unittest.skipUnless(shutil.which("nasm"), "nasm is required")
    def test_boot_sector_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "boot.bin"
            subprocess.run(["nasm", "-f", "bin", "boot.asm", "-o", str(output)],
                           cwd=ROOT, check=True)
            self.assertEqual(output.read_bytes(), (MODEL / "boot.bin").read_bytes())


if __name__ == "__main__":
    unittest.main()
