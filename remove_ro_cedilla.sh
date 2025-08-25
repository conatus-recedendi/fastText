#!/usr/bin/env bash
# normalize_ro.sh
# - ş/Ş (U+015F/U+015E) -> ș/Ș (U+0219/U+0218)
# - ţ/Ţ (U+0163/U+0162) -> ț/Ț (U+021B/U+021A)
# - NFC 정규화
# 사용:
#   cat in.txt | ./normalize_ro.sh > out.txt
#   ./normalize_ro.sh -i file1.txt file2.txt   # 인플레이스 수정(.bak 백업)

set -euo pipefail

inplace=0
if [[ "${1:-}" == "-i" ]]; then
  inplace=1
  shift
fi

normalize_cmd='
  use utf8;
  binmode(STDIN,  ":utf8");
  binmode(STDOUT, ":utf8");
  binmode(STDERR, ":utf8");
  while (read(STDIN, my $buf, 1 << 20)) {   # 1MB 씩 처리(대용량 안전)
    $buf =~ s/\x{015F}/\x{0219}/g;  # ş -> ș
    $buf =~ s/\x{015E}/\x{0218}/g;  # Ş -> Ș
    $buf =~ s/\x{0163}/\x{021B}/g;  # ţ -> ț
    $buf =~ s/\x{0162}/\x{021A}/g;  # Ţ -> Ț
    print $buf;
  }
'

nfc_cmd='
  use utf8;
  use Unicode::Normalize;
  binmode(STDIN,  ":utf8");
  binmode(STDOUT, ":utf8");
  while (<STDIN>) {
    print NFC($_);
  }
'

if (( inplace == 0 )); then
  # stdin -> stdout
  perl -CSDA -e "$normalize_cmd" \
  | perl -CSDA -MUnicode::Normalize -e "$nfc_cmd"
else
  # 인플레이스: 각 파일을 .bak 백업 후 교체
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 -i <files...>" >&2
    exit 1
  fi
  for f in "$@"; do
    [[ -f "$f" ]] || { echo "[WARN] skip: $f (not a file)"; continue; }
    cp -p "$f" "$f.bak"
    perl -CSDA -e "$normalize_cmd" < "$f.bak" \
      | perl -CSDA -MUnicode::Normalize -e "$nfc_cmd" > "$f.tmp"
    mv "$f.tmp" "$f"
  done
fi
