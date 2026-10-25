# Parse Rust libtest metadata; a listing is not evidence that any test executed.
[CmdletBinding(DefaultParameterSetName='Validate')]
param(
    [Parameter(Mandatory=$true,ParameterSetName='Validate')][string]$ListPath,
    [Parameter(Mandatory=$true,ParameterSetName='Validate')][int]$ExpectedCount,
    [Parameter(Mandatory=$true,ParameterSetName='Validate')][string]$ExpectedNames,
    [Parameter(Mandatory=$true,ParameterSetName='SelfTest')][switch]$SelfTest
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

function Confirm-TestList([string]$path,[int]$count,[string]$names) {
    [string[]]$expected=@($names -split '\|')
    try { $lines=[IO.File]::ReadAllLines($path) } catch {
        Write-Output ('H3_COMPILED_TEST_LIST '+(@{
            path=$path;self_test=[bool]$SelfTest;actual_count=$null;expected_count=$count;names=@()
            missing=$expected;extra=@();read_error=$_.Exception.Message
        }|ConvertTo-Json -Compress -Depth 4))
        throw
    }
    # Rust writes LF on Windows too. ReadAllLines handles LF and CRLF. Cargo stderr
    # may share the file; only complete libtest name records count, never diagnostics.
    [string[]]$actual=@(foreach ($line in $lines) {
        if ($line -cmatch '^([A-Za-z0-9_:]+): test$') { $Matches[1] }
    })
    $actualSet=[System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $expectedSet=[System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in $actual) { $null=$actualSet.Add($name) }
    foreach ($name in $expected) { $null=$expectedSet.Add($name) }
    [string[]]$missing=@($expected | Where-Object { -not $actualSet.Contains($_) })
    [string[]]$extra=@($actual | Where-Object { -not $expectedSet.Contains($_) })
    Write-Output ('H3_COMPILED_TEST_LIST '+(@{
        path=$path;self_test=[bool]$SelfTest;actual_count=$actual.Count;expected_count=$count;names=$actual
        expected_names=$expected;missing=$missing;extra=$extra
    }|ConvertTo-Json -Compress -Depth 4))
    if ($count -lt 1 -or $expected.Count -ne $count) {
        throw 'H3 compiled inventory expected a positive count matching its name guards'
    }
    foreach ($name in $expected) {
        if ($name -cnotmatch '^[A-Za-z0-9_:]+$') { throw 'H3 compiled inventory has an invalid expected test name' }
    }
    if ($actual.Count -ne $count) {
        throw "H3 compiled inventory count mismatch: actual=$($actual.Count) expected=$count"
    }
    if ($actualSet.Count -ne $actual.Count -or $expectedSet.Count -ne $expected.Count) {
        throw 'H3 compiled inventory contains duplicate test names'
    }
    if ($missing.Count -ne 0 -or $extra.Count -ne 0) {
        throw "H3 compiled inventory name mismatch: missing=$($missing -join '|') extra=$($extra -join '|')"
    }
}

function Invoke-InventorySelfTest([scriptblock]$validate) {
    $root=Join-Path ([IO.Path]::GetTempPath()) ('h3-test-list-'+[guid]::NewGuid().ToString('N'))
    $null=[IO.Directory]::CreateDirectory($root)
    $encoding=[Text.UTF8Encoding]::new($false)
    try {
        foreach ($ending in @("`n","`r`n")) {
            $path=Join-Path $root 'inventory.list'
            $noise=@('   Compiling candle-gen v0.0.0','Running tests\main.rs (target\debug\integration.exe)','Finished test profile')
            $valid=@($noise)+@('fixture::b: test','fixture::a: test','2 tests, 0 benchmarks')
            [IO.File]::WriteAllText($path,($valid -join $ending)+$ending,$encoding)
            & $validate $path 2 'fixture::a|fixture::b'
            $invalid=@(
                @{label='zero';rows=@($noise);count=2;names='fixture::a|fixture::b'},
                @{label='partial';rows=@($noise)+@('fixture::a: test');count=2;names='fixture::a|fixture::b'},
                @{label='extra';rows=@($valid)+@('fixture::c: test');count=2;names='fixture::a|fixture::b'},
                @{label='missing';rows=@('fixture::a: test','fixture::c: test');count=2;names='fixture::a|fixture::b'},
                @{label='wrong-namespace';rows=@('dit::model::veda_controls_tests::control: test');count=1;names='dit::model::tests::control'},
                @{label='substring';rows=@('otherfixture::a: test','fixture::b: test');count=2;names='fixture::a|fixture::b'},
                @{label='trailing';rows=@('fixture::a_extra: test','fixture::b: test');count=2;names='fixture::a|fixture::b'},
                @{label='case';rows=@('Fixture::a: test','fixture::b: test');count=2;names='fixture::a|fixture::b'},
                @{label='duplicate';rows=@('fixture::a: test','fixture::a: test');count=2;names='fixture::a|fixture::a'},
                @{label='nonpositive';rows=@($valid);count=0;names='fixture::a|fixture::b'}
            )
            foreach ($case in $invalid) {
                [IO.File]::WriteAllText($path,($case.rows -join $ending)+$ending,$encoding)
                $rejected=$false
                try { & $validate $path $case.count $case.names } catch {
                    if ($_.Exception.Message -notlike 'H3 compiled inventory*') { throw }
                    $rejected=$true
                }
                if (-not $rejected) { throw "H3 compiled list self-test accepted $($case.label) metadata" }
            }
        }
        $readFailed=$false
        try { & $validate (Join-Path $root 'absent.list') 2 'fixture::a|fixture::b' } catch {
            $readException=$_.Exception
            while ($null -ne $readException.InnerException) { $readException=$readException.InnerException }
            if ($readException -isnot [IO.FileNotFoundException]) { throw }
            $readFailed=$true
        }
        if (-not $readFailed) { throw 'H3 compiled list self-test accepted an unreadable list' }
    } finally { [IO.Directory]::Delete($root,$true) }
}

if ($SelfTest) {
    $correct={param($path,$count,$names) Confirm-TestList $path $count $names}
    Invoke-InventorySelfTest $correct
    # A behavioral mutation that ignores a missing-name error must make the SAME
    # self-test harness fail. It changes no production function or repository file.
    $ignoreMissing={
        param($path,$count,$names)
        try { Confirm-TestList $path $count $names } catch {
            if ($_.Exception.Message -like 'H3 compiled inventory name mismatch*') { return }
            throw
        }
    }
    $mutantRejected=$false
    try { Invoke-InventorySelfTest $ignoreMissing } catch {
        if ($_.Exception.Message -notlike 'H3 compiled list self-test accepted missing metadata') { throw }
        $mutantRejected=$true
    }
    if (-not $mutantRejected) { throw 'H3 compiled list self-test failed to reject its missing-name mutant' }
    Write-Output 'H3_COMPILED_TEST_LIST_MUTANT_REJECTED missing-name'
    Invoke-InventorySelfTest $correct
    Write-Output 'H3_COMPILED_TEST_LIST_SELF_TEST passed: LF, CRLF, Cargo noise, ten invalid selection classes and unreadable input'
    exit 0
}

Confirm-TestList $ListPath $ExpectedCount $ExpectedNames
