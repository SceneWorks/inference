"""Source-only selector/argv discrimination; no selector, Cargo, model or shell is executed."""
from pathlib import Path
import tempfile
import unittest

from scripts import check_model_inference_policy as policy

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = '''import argparse
import fnmatch

def choose(scope_value: str, specs: list[str]) -> list[str]:
    if scope_value.strip() in ("", "all"):
        return [arg for spec in specs for arg in ("-p", spec)]
    names = sorted(name for name in scope_value.split()
                   if any(fnmatch.fnmatchcase(name, spec) for spec in specs))
    return [arg for name in names for arg in ("-p", name)]

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("select")
    command.add_argument("--scope", default="all")
    command.add_argument("-p", dest="specs", action="append", required=True)
    args = parser.parse_args()
    if args.command == "select":
        print(" ".join(choose(args.scope, args.specs)))
        return 0
    raise ValueError("unused producer")
if __name__ == "__main__":
    main()
'''


class PackageSelectorSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'tools').mkdir()
        self.selector = self.root / 'tools/arbitrary_selector.py'
        self.selector.write_text(SELECTOR, encoding='utf-8')
        for name in ('check_compiled_model_ignores.py', 'run_ci_collection.py'):
            (self.root / 'tools' / name).write_text((ROOT / 'scripts/ci' / name).read_text(encoding='utf-8'), encoding='utf-8')

    def errors(self, *, mutate_array='', collection=False):
        wrapper = 'python3 tools/run_ci_collection.py -- ' if collection else ''
        source = '''args=$(python3 tools/arbitrary_selector.py select --scope "$TEST_SCOPE" -p core-llm -p sceneworks-gen-core)
read -r -a packages <<< "$args"
if [[ ${#packages[@]} -eq 0 ]]; then
    echo "No package selected"
    exit 0
fi
''' + mutate_array + '\npython3 -I tools/check_compiled_model_ignores.py -- ' + wrapper + 'cargo test --locked --lib "${packages[@]}"\n'
        # The collection helper recognizes this repository-relative controller spelling itself.
        source = source.replace('tools/run_ci_collection.py', 'scripts/ci/run_ci_collection.py')
        if collection:
            target = self.root / 'scripts/ci'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'run_ci_collection.py').write_text((ROOT / 'scripts/ci/run_ci_collection.py').read_text(encoding='utf-8'), encoding='utf-8')
        graph = policy.RunGraph(self.root)
        graph.shell(source, ('renamable-source-fixture',), self.root)
        return [detail for kind, detail in graph.findings if kind != 'ordinary']

    def test_declared_package_universe_crosses_read_array_and_compiled_ignore(self):
        self.assertEqual(self.errors(), [])

    def test_collection_wrapper_retains_the_same_bounded_package_universe(self):
        self.assertEqual(self.errors(collection=True), [])

    def test_removing_match_filter_cannot_authorize_dynamic_cargo_flags(self):
        self.selector.write_text(SELECTOR.replace('if any(fnmatch.fnmatchcase(name, spec) for spec in specs)', 'if True'), encoding='utf-8')
        self.assertTrue(self.errors())

    def test_selector_cannot_emit_an_ignore_override(self):
        self.selector.write_text(SELECTOR.replace('("-p", spec)', '("--ignored", spec)'), encoding='utf-8')
        self.assertTrue(self.errors())

    def test_word_split_must_exclude_whitespace_inside_package_tokens(self):
        self.selector.write_text(SELECTOR.replace('scope_value.split()', 'scope_value.split("|")'), encoding='utf-8')
        self.assertTrue(self.errors())

    def test_rebound_matcher_cannot_certify_package_names(self):
        self.selector.write_text(SELECTOR + '\nfnmatch = object()\n', encoding='utf-8')
        self.assertTrue(self.errors())

    def test_array_mutation_cannot_reuse_pre_mutation_selector_proof(self):
        for mutation in ('packages+=(--ignored)', 'packages=--ignored', 'unset packages', 'packages[0]=--ignored'):
            with self.subTest(mutation=mutation):
                self.assertTrue(self.errors(mutate_array=mutation))

    def test_filtered_local_mutation_cannot_keep_package_domain_proof(self):
        for mutation in ('names += ["--ignored"]', 'names[0] = "--ignored"'):
            with self.subTest(mutation=mutation):
                self.selector.write_text(SELECTOR.replace('    return [arg for name in names', '    ' + mutation + '\n    return [arg for name in names'), encoding='utf-8')
                self.assertTrue(self.errors())

    def test_argparse_package_values_are_not_trusted_after_mutation(self):
        self.selector.write_text(SELECTOR.replace('    if args.command == "select":', '    args.specs.append("--ignored")\n    if args.command == "select":'), encoding='utf-8')
        self.assertTrue(self.errors())

    def test_extra_stdout_cannot_be_dropped_from_selector_proof(self):
        self.selector.write_text(SELECTOR.replace('        print(" ".join', '        print("--ignored")\n        print(" ".join'), encoding='utf-8')
        self.assertTrue(self.errors())

    def test_selector_side_effect_is_traced_even_with_a_bounded_stdout(self):
        self.selector.write_text(SELECTOR.replace('import argparse', 'import argparse\nimport subprocess\nsubprocess.run(["cargo", "run", "--bin", "model"])'), encoding='utf-8')
        self.assertTrue(any('Cargo native driver' in error for error in self.errors()))


if __name__ == '__main__':
    unittest.main()
