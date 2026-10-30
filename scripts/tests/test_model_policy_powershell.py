"""Source-only PowerShell edge tests. These never invoke PowerShell or a model."""
import os
from pathlib import Path
import tempfile
import unittest

from scripts.check_model_inference_policy import RunGraph
from scripts.ci.model_policy_powershell import analyze

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get('MODEL_POLICY_POWERSHELL_SOURCE_ROOT', ROOT))

MOCK = '''using System;
using System.IO;
using System.Threading;
public static class PipeFixture {
    public static int Main(string[] args) {
        byte[] block = new byte[65536];
        var stdout = Console.OpenStandardOutput();
        stdout.Write(block, 0, block.Length);
        var deadline = DateTime.UtcNow.AddSeconds(60);
        while (!File.Exists(args[args.Length - 1])) {
            if (DateTime.UtcNow > deadline) return 91;
            Thread.Sleep(25);
        }
        return 0;
    }
}
'''

SUPERVISOR = '''
function Run-Child([string]$exe, [string[]]$arguments, [scriptblock]$check) {
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=$exe
    $start.Arguments=[string]::Join(' ',$arguments)
    $process=[Diagnostics.Process]::new()
    $process.StartInfo=$start
    $started=$process.Start()
    & $check @(1)
}
'''


class PowerShellExecutionEdges(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def trace(self, text, args=(), root=None):
        graph = RunGraph(root or self.root)
        analyze(graph, text, ('test entry',), root or self.root, args)
        return graph.findings

    def bad(self, text, args=(), root=None):
        return [(kind, detail) for kind, detail in self.trace(text, args, root)
                if kind in ('model', 'proof', 'unknown')]

    def assert_safe(self, text, args=(), root=None):
        self.assertEqual(self.bad(text, args, root), [])

    def assert_rejected(self, text, args=(), root=None, kind=None):
        findings = self.bad(text, args, root)
        self.assertTrue(findings, text)
        if kind:
            self.assertTrue(any(k == kind for k, _ in findings), findings)
        return findings

    def test_literal_metadata_and_nested_tool_queries(self):
        self.assert_safe('''
$ErrorActionPreference='Stop'
$default=@(& rustup default)
$version=(& cargo --version)
@{source=(git rev-parse HEAD);cargo=$version}|ConvertTo-Json|Set-Content source.json
Add-Content -Path cache.txt -Value 'data'
Remove-Item cache.txt -Force
''')

    def test_declared_switch_selects_source_branch_and_early_exit(self):
        source = '''param([switch]$SelfTest)
if ($SelfTest) { Write-Output 'metadata'; exit 0 }
& target/debug/model-driver --generate
'''
        self.assert_safe(source, ('-SelfTest',))
        self.assert_rejected(source, kind='model')

    def test_undeclared_switch_cannot_hide_model_entry(self):
        source = '''if ($SelfTest) { exit 0 }
& target/debug/model-driver --generate
'''
        findings = self.assert_rejected(source, ('-SelfTest',), kind='model')
        self.assertTrue(any('undeclared' in detail for _, detail in findings))

    def test_selected_selftest_injected_model_is_rejected(self):
        source = '''param([switch]$SelfTest)
if ($SelfTest) { & cargo run --bin learned_layer; exit 0 }
'''
        self.assert_rejected(source, ('-SelfTest',), kind='model')

    def test_selected_selftest_injected_proof_is_rejected(self):
        source = '''param([switch]$SelfTest)
if ($SelfTest) { python scripts/ci/model_inference_proofs.py --execute; exit 0 }
'''
        self.assert_rejected(source, ('-SelfTest',), kind='proof')

    def test_unknown_conditional_exit_does_not_hide_later_model(self):
        self.assert_rejected('if ($missing) { exit 0 }; & target/model', kind='model')

    def test_condition_negation_does_not_invert_logical_precedence(self):
        self.assert_rejected('if (-not $false -or $true) { & target/model }', kind='model')

    def test_unknown_branch_variable_join_cannot_overwrite_unsafe_path(self):
        self.assert_rejected("if ($missing) { $exe='target/model' } else { $exe='echo' }; & $exe")

    def test_unknown_branch_function_join_retains_unsafe_definition(self):
        self.assert_rejected('''
if ($missing) { function Worker { & target/model } } else { function Worker { echo ok } }
Worker
''', kind='model')

    def test_exception_branch_join_cannot_hide_success_path_model(self):
        self.assert_rejected("""
$exe='echo'
try { $file=[IO.File]::Open('metadata'); $exe='target/model' }
catch { $exe='echo' }
& $exe
""")

    def test_cmd_single_separator_is_not_part_of_set_metadata(self):
        self.assert_rejected("""
$path=Join-Path ([IO.Path]::GetTempPath()) 'any.bat'
[IO.File]::WriteAllText($path,'set VALUE=metadata & python model_inference_proofs.py')
& $env:ComSpec @('/d','/c',('""'+$path+'" >nul && set"'))
""", kind='proof')

    def test_recursive_function_and_scriptblock_fail_closed(self):
        self.assert_rejected('function Recurse { Recurse }; Recurse', kind='unknown')
        self.assert_rejected('$loop={& $loop}; & $loop', kind='unknown')

    def test_dynamic_variable_invocation_is_unknown(self):
        self.assert_rejected('& $env:MODEL_TOOL --tiny --random-weights', kind='unknown')

    def test_unknown_bare_command_is_not_metadata(self):
        self.assert_rejected('Invoke-LearnedLayer -Device cpu -Tiny', kind='unknown')

    def test_unknown_method_and_type_fail_closed(self):
        for source in ('$model.Forward()', '[LearnedLayer]::new()', '[LearnedLayer]::Forward()',
                       '[void][LearnedLayer]::Forward()', '$data.Copy().Forward()'):
            with self.subTest(source=source):
                self.assert_rejected(source, kind='unknown')

    def test_metadata_arguments_recursively_trace_execution(self):
        for source in ('Write-Output (python model_inference_proofs.py)',
                       'Write-Output "$(python model_inference_proofs.py)"',
                       '@{key=(python model_inference_proofs.py)}|ConvertTo-Json',
                       'Get-Content file | ForEach-Object { python model_inference_proofs.py }'):
            with self.subTest(source=source):
                self.assert_rejected(source, kind='proof')

    def test_literal_strings_and_comments_are_not_execution(self):
        self.assert_safe("# python model_inference_proofs.py\nWrite-Output 'python model_inference_proofs.py'")

    def test_function_and_callback_bind_actual_arguments(self):
        source = '''
function Dispatch([string]$exe, [string[]]$argv, [scriptblock]$callback) {
    & $exe @argv
    & $callback
}
$callback={python model_inference_proofs.py}
Dispatch cargo @('--version') $callback
'''
        self.assert_rejected(source, kind='proof')

    def test_source_function_named_native_is_not_exempted(self):
        self.assert_rejected('function Native { & target/model }; Native', kind='model')

    def test_source_function_named_selftest_is_not_exempted(self):
        self.assert_rejected('function Invoke-SelfTest { python model_inference_proofs.py }; Invoke-SelfTest', kind='proof')

    def test_process_startinfo_resolves_executable_and_arguments(self):
        self.assert_safe(SUPERVISOR + "Run-Child cargo @('--version') {param($ids) Write-Output $ids}")
        self.assert_rejected(SUPERVISOR + "Run-Child cargo @('run','--bin','model') {}", kind='model')
        self.assert_rejected(SUPERVISOR + "Run-Child $unknown @('--version') {}", kind='unknown')

    def test_startprocess_resolves_arguments(self):
        self.assert_safe("$p=Start-Process -FilePath cargo -ArgumentList @('--version') -PassThru")
        self.assert_rejected("Start-Process -FilePath cargo -ArgumentList @('run','--bin','model')", kind='model')
        self.assert_rejected("Start-Process -FilePath $unknown -ArgumentList @('--version')", kind='unknown')

    def test_process_static_start_resolves_arguments(self):
        self.assert_rejected("[Diagnostics.Process]::Start('cargo','run --bin model')", kind='model')

    def test_unknown_process_startinfo_is_rejected(self):
        self.assert_rejected('$p=[Diagnostics.Process]::new();$p.Start()', kind='unknown')

    def test_generated_powershell_uses_written_source_with_any_filename(self):
        self.assert_safe('''
$path=Join-Path ([IO.Path]::GetTempPath()) 'arbitrary.ps1'
[IO.File]::WriteAllText($path,"Start-Sleep -Milliseconds 1`nexit 0")
Start-Process -FilePath powershell -ArgumentList @('-NoProfile','-File',$path)
''')

    def test_generated_powershell_mutation_is_rejected(self):
        self.assert_rejected('''
$path=Join-Path ([IO.Path]::GetTempPath()) 'arbitrary.ps1'
[IO.File]::WriteAllText($path,'python model_inference_proofs.py')
Start-Process -FilePath powershell -ArgumentList @('-NoProfile','-File',$path)
''', kind='proof')

    def test_generated_path_separators_preserve_exact_source_identity(self):
        source = r'''
[IO.File]::WriteAllText('C:/fixture/child.ps1','Write-Output metadata')
& powershell -File 'C:\fixture\child.ps1'
'''
        self.assert_safe(source)
        self.assert_rejected(source.replace('Write-Output metadata', 'python model_inference_proofs.py'), kind='proof')
        self.assert_rejected(source.replace("-File 'C:\\fixture\\", "-File 'C:\\different\\"), kind='unknown')

    def test_source_generated_grandchild_is_traced(self):
        source = '''
$root=Join-Path ([IO.Path]::GetTempPath()) 'chain'
$child=Join-Path $root 'child.ps1'
$parent=Join-Path $root 'parent.ps1'
$body=@'
Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList @('-File',(Join-Path $PSScriptRoot 'child.ps1'))
'@
[IO.File]::WriteAllText($child,'python model_inference_proofs.py')
[IO.File]::WriteAllText($parent,$body)
& powershell -File $parent
'''
        self.assert_rejected(source, kind='proof')

    def test_generated_unknown_branch_cannot_overwrite_unsafe_source(self):
        source = '''
$path=Join-Path ([IO.Path]::GetTempPath()) 'child.ps1'
if ($unknown) { [IO.File]::WriteAllText($path,'python model_inference_proofs.py') }
else { [IO.File]::WriteAllText($path,'echo ok') }
& powershell -File $path
'''
        self.assert_rejected(source)
        self.assert_rejected(source.replace('}\nelse', '} else'), kind='proof')

    def test_generated_cmd_source_is_inspected(self):
        source = '''
$path=Join-Path ([IO.Path]::GetTempPath()) 'any.bat'
[IO.File]::WriteAllText($path,"@echo off`r`nset `"VALUE=data`"`r`nexit /b 0`r`n")
& $env:ComSpec @('/d','/c',('""'+$path+'" >nul && set"'))
'''
        self.assert_safe(source)
        self.assert_rejected(source.replace('exit /b 0', 'python model_inference_proofs.py'), kind='proof')

    def test_cmd_literal_exit_is_safe_but_dynamic_command_is_unknown(self):
        self.assert_safe("foreach ($code in @(0,7)) { Start-Process -FilePath $env:ComSpec -ArgumentList @('/d','/c',('exit '+$code)) }")
        self.assert_rejected("Start-Process -FilePath $env:ComSpec -ArgumentList @('/c',$env:COMMAND)", kind='unknown')

    def generated_mock(self, body=MOCK, tail='& $exe'):
        return "$exe=Join-Path ([IO.Path]::GetTempPath()) 'any-name.exe'\nAdd-Type -OutputAssembly $exe -OutputType ConsoleApplication -TypeDefinition @'\n" + body + "'@\n" + tail

    def test_generated_native_mock_requires_inspected_source(self):
        self.assert_safe(self.generated_mock())
        self.assert_safe(self.generated_mock(tail="Copy-Item -LiteralPath $exe -Destination 'renamed.exe'\n& renamed.exe"))

    def test_generated_native_unknown_source_is_not_a_mock(self):
        self.assert_rejected("Add-Type -OutputAssembly arbitrary.exe -TypeDefinition $env:CODE; & arbitrary.exe", kind='unknown')

    def test_generated_native_mock_model_launch_mutation_is_rejected(self):
        mutation = MOCK.replace('return 0;', 'System.Diagnostics.Process.Start("target/model"); return 0;')
        self.assert_rejected(self.generated_mock(mutation), kind='unknown')

    def test_generated_native_mock_unknown_calls_and_constructors_rejected(self):
        for mutation in ('Model.Forward();', 'var x = new Model();', 'Helper.Run();',
                         'System.Reflection.Assembly.Load("model");', 'dynamic x = args; x.Forward();'):
            with self.subTest(mutation=mutation):
                self.assert_rejected(self.generated_mock(MOCK.replace('return 0;', mutation + ' return 0;')), kind='unknown')

    def test_external_cargo_subcommand_requires_generated_source_and_path(self):
        source = self.generated_mock(tail='''
$folder=Split-Path $exe
& cargo fixture
''')
        self.assert_rejected(source, kind='unknown')
        source = source.replace("'any-name.exe'", "'cargo-fixture.exe'").replace('$folder=Split-Path $exe', "$env:PATH=([IO.Path]::GetTempPath())+';'+$env:PATH")
        self.assert_safe(source)
        self.assert_rejected(source.replace('stdout.Write(block, 0, block.Length);', 'Model.Forward();'), kind='unknown')

    def test_repository_script_uses_source_and_switch_arguments(self):
        (self.root / 'entry.ps1').write_text('param([switch]$Metadata)\nif ($Metadata) { echo ok;exit 0 };python model_inference_proofs.py', encoding='utf-8')
        self.assert_safe('& powershell -File ./entry.ps1 -Metadata')
        self.assert_rejected('& powershell -File ./entry.ps1', kind='proof')

    def test_source_outside_repository_is_unresolved(self):
        self.assert_rejected('& powershell -File /tmp/unavailable.ps1', kind='unknown')

    def test_dynamic_eval_and_malformed_source_fail_closed(self):
        for source in ('Invoke-Expression $env:CODE', '&', '$x="unterminated', 'if ($true) {'):
            with self.subTest(source=source):
                self.assert_rejected(source, kind='unknown')

    def test_source_selftest_paths_and_mutations(self):
        for filename in ('moss-native-terminal.ps1', 'h3-native-test-list.ps1'):
            path = SOURCE_ROOT / 'scripts/ci' / filename
            if not path.is_file():
                continue
            with self.subTest(filename=filename):
                source = path.read_text(encoding='utf-8')
                self.assert_safe(source, ('-SelfTest',), SOURCE_ROOT)
                self.assert_rejected(source.replace('if ($SelfTest) {', 'if ($SelfTest) {\npython model_inference_proofs.py', 1), ('-SelfTest',), SOURCE_ROOT, kind='proof')
                if 'Console.OpenStandardOutput()' in source:
                    mutant = source.replace('Console.OpenStandardOutput()', 'System.Diagnostics.Process.Start("target/model")', 1)
                    self.assert_rejected(mutant, ('-SelfTest',), SOURCE_ROOT, kind='unknown')


if __name__ == '__main__':
    unittest.main()
