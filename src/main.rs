use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{env, fs, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    dotenvy::dotenv().ok();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let is_local = env::var("local")
        .map(|local| local == "true")
        .unwrap_or(false);

    let model = if is_local {
        "nvidia/nemotron-3-super-120b-a12b:free"
    } else {
        "anthropic/claude-haiku-4.5"
    };

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let mut msgs: Vec<Message> = vec![Message {
        role: "user".to_string(),
        content: Some(args.prompt),
        tool_call_id: None,
        tool_calls: None,
    }];

    let mut resp: LLMResponse;

    loop {
        let vr = json!({
            "messages": msgs,
            "model": model,
            "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                      "type": "object",
                      "properties": {
                        "file_path": {
                          "type": "string",
                          "description": "The path to the file to read"
                        }
                      },
                      "required": ["file_path"]
                    }
                  }
                }
            ]
        });

        #[allow(unused_variables)]
        let response: Value = client.chat().create_byot(vr).await?;

        resp = serde_json::from_value(response)?;

        let Some(choice) = resp.choices.first() else {
            break;
        };

        let finish_reason = choice.finish_reason.clone();
        let ast_msg = choice.message.clone();
        msgs.push(ast_msg.clone());

        let mut processed_tcs = false;

        if let Some(tcs) = ast_msg.extract_toolcalls() {
            // if !tcs.is_empty() {
                for tc in tcs {
                    if let Some(path) = tc.extract_filepath() {
                        let fp: FilePath = serde_json::from_str(path)?;
                        let ctn = fs::read_to_string(fp.file_path.as_str())?;
    
                        msgs.push(Message {
                            role: "tool".to_string(),
                            content: Some(ctn),
                            tool_call_id: Some(tc.id.to_string()),
                            tool_calls: None,
                        });

                        processed_tcs = true;
                    }
                }
            
            }

            if processed_tcs {
                continue;
            }
        // }
       
        if let Some(content) = ast_msg.content {
            println!("{content}");
            break;
        }

        if finish_reason.as_deref() == Some("stop") {
            break;
        }

        break;
    }

    // You can use print statements as follows for debugging, they'll be visible when running tests.
    eprintln!("Logs from your program will appear here!");

    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct FilePath {
    file_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(unused)]
struct LLMResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(unused)]
struct Choice {
    index: i32,
    message: Message,
    finish_reason: Option<String>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(unused)]
struct Message {
    role: String,
    content: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(unused)]
struct ToolCall {
    id: String,
    index: i32,
    function: Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(unused)]
struct Function {
    name: String,
    arguments: String,
}

impl Message {
    fn extract_toolcalls(&self) -> Option<&Vec<ToolCall>> {
        self.tool_calls.as_ref()
    }
}

impl ToolCall {
    fn extract_filepath(&self) -> Option<&str> {
        let fct = &self.function;
        if fct.name.as_str() == "Read" {
            return Some(fct.arguments.as_str());
        }
        None
    }
}

// fn extract_filepath(choice: &Choice) -> Option<&str> {
//     if let Some(tool_calls) = &choice.message.extract_toolcalls() {
//         let fct = &tool_calls[0].function;
//         if fct.name == "Read" {
//             return Some(fct.arguments.as_str());
//         }
//     }

//     return None;
// }
