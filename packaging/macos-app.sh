#!/bin/sh
# packs the macOS binary as Zakuro.app, with MoltenVK inside so that the 3D
# draws on the GPU with nothing else installed, and zips it.
#
#   macos-app.sh <binary> <version> <output.zip>
set -eu

moltenvk_url=https://github.com/KhronosGroup/MoltenVK/releases/download/v1.4.2/MoltenVK-macos.tar
moltenvk_sha256=f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e

binary=$1
version=$2
output=$(pwd)/$3
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

curl -fsSL -o "$work/moltenvk.tar" "$moltenvk_url"
echo "$moltenvk_sha256  $work/moltenvk.tar" | shasum -a 256 -c -
tar xf "$work/moltenvk.tar" -C "$work" MoltenVK/LICENSE MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib

contents=$work/Zakuro.app/Contents
mkdir -p "$contents/MacOS" "$contents/Resources" "$contents/Frameworks"
cp "$binary" "$contents/MacOS/zakuro"
cp "$work/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib" "$contents/Frameworks/"
cp "$work/MoltenVK/LICENSE" "$contents/Resources/MoltenVK-LICENSE"
cp "$here/Info.plist" "$contents/Info.plist"
/usr/libexec/PlistBuddy \
    -c "Set :CFBundleShortVersionString $version" \
    -c "Set :CFBundleVersion $version" \
    "$contents/Info.plist"

# the icon at every size macOS asks for, each with its double
icons=$work/zakuro.iconset
mkdir "$icons"
for size in 16 32 128 256; do
    sips -z "$size" "$size" "$here/zakuro.png" --out "$icons/icon_${size}x${size}.png" > /dev/null
    double=$((size * 2))
    sips -z "$double" "$double" "$here/zakuro.png" --out "$icons/icon_${size}x${size}@2x.png" > /dev/null
done
cp "$here/zakuro.png" "$icons/icon_512x512.png"
iconutil -c icns "$icons" -o "$contents/Resources/zakuro.icns"

# signed for no one, which is what lets it run on Apple silicon at all
codesign --force --deep --sign - "$work/Zakuro.app"
ditto -c -k --keepParent "$work/Zakuro.app" "$output"
