use csv::ReaderBuilder;
use std::{fs, path::Path};


pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
pub enum DatasetType {
    JSON,
    CSV,
}

impl Dataset {
    pub fn new(
        pretraining_data_path: String,
        chat_training_data_path: String,
        type_of_data: DatasetType,
    ) -> Self {
        let pretraining_data: Vec<String>;
        let chat_training_data: Vec<String>;

        match type_of_data {
            DatasetType::CSV => {
                pretraining_data = get_data_from_csv(pretraining_data_path);
                chat_training_data = get_data_from_csv(chat_training_data_path);
            }
            DatasetType::JSON => {
                pretraining_data = get_data_from_json(pretraining_data_path);
                chat_training_data = get_data_from_json(chat_training_data_path);
            }
        }

        Dataset {
            pretraining_data: pretraining_data.clone(),
            chat_training_data: chat_training_data.clone(),
        }
    }
}

fn get_data_from_json(path: String) -> Vec<String> {
    // convert json file to Vec<String>
    let data_json = fs::read_to_string(path).expect("Failed to read data file");
    let data: Vec<String> = serde_json::from_str(&data_json).expect("Failed to parse data file");
    data
}

//===================================================================
//  Funktionsname : get_data_from_csv
//  Kurzbeschreibung:
//      Liest eine CSV-Datei zeilenweise ein, prüft das Terminator-
//      Token "</s>" und ergänzt es bei Bedarf automatisch.
//      Liefert einen Vektor validierter Datensätze.
//
//  Autor      : Marcus Schlieper
//  Historie   :
//      1.0 | 23.11.2025 | MS | Erste Implementierung
//      1.2 | 23.11.2025 | MS | Automatische Token-Ergänzung
//===================================================================

pub fn get_data_from_csv(path: String) -> Vec<String> {
    // Konstante für das Terminator-Token
    const S_TERMINATOR: &str = "</s>";

    // Datei öffnen; bei Fehler mit klarer Meldung beenden
    let file_path = Path::new(&path);
    let file = fs::File::open(file_path)
        .unwrap_or_else(|e| panic!("CSV-Datei {:?} konnte nicht geöffnet werden: {}", file_path, e));

    // CSV-Reader ohne Kopfzeile konfigurieren
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    // Ergebnisvektor initialisieren
    let mut v_data: Vec<String> = Vec::new();

    // Über alle Datensätze iterieren
    for (_HIDDEN_DIM, result) in rdr.records().enumerate() {
        let record = result.expect("Fehler beim Lesen eines CSV-Datensatzes");

        // Spalten zu einer einzelnen Zeile zusammenfügen
        let mut s_line: String = record.iter().collect::<Vec<&str>>().join(",");

        // Überflüssige Leerzeichen am Zeilenende entfernen
        s_line = s_line.trim_end().to_string();

        // Prüfen und ggf. Ergänzen des Terminator-Tokens
        if !s_line.ends_with(S_TERMINATOR) {
            s_line.push_str(S_TERMINATOR);
            // Optionales Logging:
            // eprintln!("Hinweis: Terminator in Zeile {} ergänzt.", i_index + 1);
        }

        // Validierte Zeile dem Ergebnisvektor hinzufügen
        v_data.push(s_line);
    }

    v_data
}