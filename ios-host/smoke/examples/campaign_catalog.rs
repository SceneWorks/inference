//! Weights-free catalog export. Does not create arrays or initialize an accelerator.
fn main() {
    match ios_smoke::campaign_catalog_report() {
        Ok(catalog) => println!("{catalog}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
