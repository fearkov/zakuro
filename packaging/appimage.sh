#!/bin/sh
# packs the Linux binary as an AppImage, which holds nothing but the program:
# the libraries it needs are the ones every desktop system has, and the
# sound one has to be the system's own for its plugins to work.
#
#   appimage.sh <binary> <output>
set -eu

tool_url=https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage
tool_sha256=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0
runtime_url=https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64
runtime_sha256=2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d

binary=$1
output=$2
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

curl -fsSL -o "$work/appimagetool" "$tool_url"
curl -fsSL -o "$work/runtime" "$runtime_url"
echo "$tool_sha256  $work/appimagetool" | sha256sum -c -
echo "$runtime_sha256  $work/runtime" | sha256sum -c -
chmod +x "$work/appimagetool"

app=$work/Zakuro.AppDir
mkdir -p "$app/usr/bin"
cp "$binary" "$app/usr/bin/zakuro"
ln -s usr/bin/zakuro "$app/AppRun"
cp "$here/zakuro.desktop" "$here/zakuro.png" "$app/"

# the update information has AppImage updaters, Gear Lever's say, follow the
# releases
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 "$work/appimagetool" \
    --runtime-file "$work/runtime" \
    --updateinformation "gh-releases-zsync|fearkov|zakuro|latest|Zakuro-x86_64.AppImage.zsync" \
    "$app" "$output"
