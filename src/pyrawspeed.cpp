#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "RawSpeed-API.h"
#include "decoders/RawDecoderException.h"
#include "io/FileIOException.h"
#include "io/IOException.h"
#include "metadata/CameraMetadataException.h"
#include "parsers/CiffParserException.h"
#include "parsers/FiffParserException.h"
#include "parsers/RawParserException.h"
#include "parsers/TiffParserException.h"

namespace nb = nanobind;
using namespace rawspeed;

// CameraMetaData wrapper.
// The shared_ptr keeps the object heap-allocated so nanobind never tries to
// copy it (Apple Clang misreports it as copy-constructible).
// Two constructors are exposed:
//   CameraMetaData()        — empty metadata (no cameras.xml needed)
//   CameraMetaData(path)    — load cameras.xml from the given path (requires pugixml)
struct CameraMetaDataHolder {
    std::shared_ptr<CameraMetaData> inner;
    CameraMetaDataHolder() : inner(std::make_shared<CameraMetaData>()) {}
    explicit CameraMetaDataHolder(const std::string& path)
        : inner(std::make_shared<CameraMetaData>(path.c_str())) {}
    bool hasCamera(const std::string& make, const std::string& model,
                   const std::string& mode = "") const {
        return inner->hasCamera(make, model, mode);
    }
};

NB_MODULE(_pyrawspeed, m) {
    m.doc() = "Python bindings for librawspeed";

    // Exceptions — parents before children
    static auto exc_rse = nb::exception<RawspeedException>(m, "RawspeedError");
    static auto exc_rde = nb::exception<RawDecoderException>(m, "RawDecoderError", exc_rse.ptr());
    static auto exc_fie = nb::exception<FileIOException>(m, "FileIOError", exc_rde.ptr());
    static auto exc_cme = nb::exception<CameraMetadataException>(m, "CameraMetadataError", exc_rse.ptr());
    static auto exc_ioe = nb::exception<IOException>(m, "RawspeedIOError", exc_rse.ptr());
    static auto exc_rpe = nb::exception<RawParserException>(m, "RawParserError", exc_rse.ptr());
    static auto exc_cpe = nb::exception<CiffParserException>(m, "CiffParserError", exc_rpe.ptr());
    static auto exc_fpe = nb::exception<FiffParserException>(m, "FiffParserError", exc_rpe.ptr());
    static auto exc_tpe = nb::exception<TiffParserException>(m, "TiffParserError", exc_rpe.ptr());

    // RawImage and supporting types

    nb::enum_<CFAColor>(m, "CFAColor")
        .value("RED",        CFAColor::RED)
        .value("GREEN",      CFAColor::GREEN)
        .value("BLUE",       CFAColor::BLUE)
        .value("CYAN",       CFAColor::CYAN)
        .value("MAGENTA",    CFAColor::MAGENTA)
        .value("YELLOW",     CFAColor::YELLOW)
        .value("WHITE",      CFAColor::WHITE)
        .value("FUJI_GREEN", CFAColor::FUJI_GREEN)
        .value("UNKNOWN",    CFAColor::UNKNOWN);

    nb::enum_<RawImageType>(m, "RawImageType")
        .value("UINT16", RawImageType::UINT16)
        .value("F32",    RawImageType::F32);

    nb::class_<ColorFilterArray>(m, "ColorFilterArray")
        .def("get_color_at", &ColorFilterArray::getColorAt,
             nb::arg("x"), nb::arg("y"))
        .def("as_string", &ColorFilterArray::asString)
        .def_prop_ro("size", [](const ColorFilterArray& cfa) {
            auto s = cfa.getSize();
            return nb::make_tuple(s.x, s.y);
        });

    nb::class_<RawImage>(m, "RawImage")
        .def_prop_ro("width",     [](const RawImage& img) { return (*img).dim.x; })
        .def_prop_ro("height",    [](const RawImage& img) { return (*img).dim.y; })
        .def_prop_ro("cpp",       [](const RawImage& img) { return (*img).getCpp(); })
        .def_prop_ro("data_type", [](const RawImage& img) { return (*img).getDataType(); })
        .def_prop_ro("is_cfa",    [](const RawImage& img) { return (*img).isCFA; })
        .def_prop_ro("cfa",       [](const RawImage& img) -> const ColorFilterArray& {
            return (*img).cfa;
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("black_level", [](const RawImage& img) { return (*img).blackLevel; })
        .def_prop_ro("white_point", [](const RawImage& img) -> nb::object {
            const auto& wp = (*img).whitePoint;
            return wp.has_value() ? nb::cast(*wp) : nb::none();
        })
        .def_prop_ro("crop_offset_x", [](const RawImage& img) { return (*img).getCropOffset().x; })
        .def_prop_ro("crop_offset_y", [](const RawImage& img) { return (*img).getCropOffset().y; })
        .def_prop_ro("make",      [](const RawImage& img) { return (*img).metadata.make; })
        .def_prop_ro("model",     [](const RawImage& img) { return (*img).metadata.model; })
        .def_prop_ro("iso_speed", [](const RawImage& img) { return (*img).metadata.isoSpeed; })
        .def_prop_ro("wb_coeffs", [](const RawImage& img) -> nb::object {
            const auto& wb = (*img).metadata.wbCoeffs;
            if (!wb.has_value())
                return nb::none();
            return nb::make_tuple((*wb)[0], (*wb)[1], (*wb)[2], (*wb)[3]);
        })
        // Zero-copy numpy view; shape is [height, width*cpp].
        .def_prop_ro("pixels", [](RawImage& img) -> nb::object {
            auto& d = *img;
            // Keeps the shared buffer alive as long as the ndarray exists.
            nb::capsule owner(new RawImage(img), [](void* p) noexcept {
                delete static_cast<RawImage*>(p);
            });
            if (d.getDataType() == RawImageType::UINT16) {
                auto arr = d.getU16DataAsCroppedArray2DRef().getAsArray2DRef();
                size_t shape[2]    = {(size_t)arr.height(), (size_t)arr.width()};
                int64_t strides[2] = {(int64_t)arr.pitch(), 1};
                return nb::cast(nb::ndarray<nb::numpy, uint16_t, nb::ndim<2>>(
                    arr[0].begin(), 2, shape, owner, strides));
            } else {
                auto arr = d.getF32DataAsCroppedArray2DRef().getAsArray2DRef();
                size_t shape[2]    = {(size_t)arr.height(), (size_t)arr.width()};
                int64_t strides[2] = {(int64_t)arr.pitch(), 1};
                return nb::cast(nb::ndarray<nb::numpy, float, nb::ndim<2>>(
                    arr[0].begin(), 2, shape, owner, strides));
            }
        });

    // CameraMetaData — wraps rawspeed's CameraMetaData.
    // CameraMetaData()       — empty, no cameras.xml needed.
    // CameraMetaData(path)   — load cameras.xml from path (requires pugixml build).
    nb::class_<CameraMetaDataHolder>(m, "CameraMetaData")
        .def(nb::init<>())
        .def(nb::init<const std::string&>(), nb::arg("path"))
        .def("has_camera", &CameraMetaDataHolder::hasCamera,
             nb::arg("make"), nb::arg("model"), nb::arg("mode") = "");

    // full decode pipeline as a single function.
    // Internally: FileReader -> Buffer -> RawParser -> RawDecoder -> RawImage.
    // `storage` owns the file buffer and lives for the duration of the call.
    m.def("decode", [](const std::string& path, const CameraMetaDataHolder& meta) {
        FileReader reader(path.c_str());
        auto [storage, buffer] = reader.readFile();

        RawParser parser(buffer);
        auto decoder = parser.getDecoder(meta.inner.get());

        // checkSupport() is intentionally omitted: it requires a populated
        // CameraMetaData (cameras.xml) and throws for unknown cameras.
        // Skipping it means unsupported-camera detection is silent, but the
        // decode attempt below will still throw if the format is unreadable.

        RawImage raw = decoder->decodeRaw();
        // decodeMetaData with an empty meta still extracts file-embedded data
        // (WB coefficients, ISO, black level, crop offset, etc.)
        decoder->decodeMetaData(meta.inner.get());

        return raw;
    }, nb::arg("path"), nb::arg("meta"));
}
