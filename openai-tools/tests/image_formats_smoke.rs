//! Guards the trimmed-down `image` feature set: `Content::from_image_file`
//! must still decode and re-encode every format it advertises.

use openai_tools::common::message::Content;

fn fixture(dir: &std::path::Path, name: &str, fmt: image::ImageFormat) -> String {
    let path = dir.join(name);
    let img = image::DynamicImage::new_rgb8(2, 2);
    img.save_with_format(&path, fmt).expect("should encode fixture");
    path.to_string_lossy().into_owned()
}

#[test]
fn from_image_file_supports_png_jpeg_gif() {
    let dir = std::env::temp_dir().join("openai_tools_image_smoke");
    std::fs::create_dir_all(&dir).unwrap();

    for (name, fmt, mime) in [
        ("a.png", image::ImageFormat::Png, "png"),
        ("a.jpg", image::ImageFormat::Jpeg, "jpg"),
        ("a.jpeg", image::ImageFormat::Jpeg, "jpeg"),
        ("a.gif", image::ImageFormat::Gif, "gif"),
    ] {
        let path = fixture(&dir, name, fmt);
        let content = Content::from_image_file(&path);

        assert_eq!(content.type_name, "input_image");
        let url = content.image_url.expect("image_url should be set");
        assert!(url.starts_with(&format!("data:image/{mime};base64,")), "wrong data URL prefix for {name}: {}", &url[..40.min(url.len())]);
        assert!(url.len() > 40, "base64 payload looks empty for {name}");
    }

    std::fs::remove_dir_all(&dir).ok();
}
