use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{env, fs, process, time::Duration};
use tokio::time::timeout;

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

    let allowed_tools = vec![
        json!({
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
        }),
        json!({
          "type": "function",
          "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
              "type": "object",
              "required": ["file_path", "content"],
              "properties": {
                "file_path": {
                  "type": "string",
                  "description": "The path of the file to write to"
                },
                "content": {
                  "type": "string",
                  "description": "The content to write to the file"
                }
              }
            }
          }
        }),
    ];

    let mut resp: LLMResponse;

    loop {
        let vr = json!({
            "messages": msgs,
            "model": model,
            "tools": allowed_tools
        });

        eprintln!(
            "Sending chat request: model={model}, messages={}, message={:#?}",
            msgs.len(),
            msgs
        );

        let response: Value =
            match timeout(Duration::from_secs(900), client.chat().create_byot(vr)).await {
                Ok(Ok(response)) => {
                    eprintln!("Received chat response");
                    response
                }
                Ok(Err(err)) => {
                    eprintln!("Chat request failed: {err:?}");
                    return Err(Box::<dyn std::error::Error>::from(err));
                }
                Err(_) => {
                    eprintln!("Chat request timed out after 15 mins");
                    return Err("Chat request timed out after 15 mins".into());
                }
            };

        eprintln!("Response: {:#?}", &response);

        resp = match serde_json::from_value(response) {
            Ok(resp) => resp,
            Err(err) => {
                eprintln!("Failed to parse chat response: {err:?}");
                return Err(Box::<dyn std::error::Error>::from(err));
            }
        };

        let Some(choice) = resp.choices.first() else {
            break;
        };

        let finish_reason = choice.finish_reason.clone();
        let ast_msg = choice.message.clone();
        msgs.push(ast_msg.clone());

        match ast_msg.extract_toolcalls() {
            Some(tcs) if !tcs.is_empty() => {
                for tc in tcs {
                    let tool_result = match tc.function_name() {
                        "Read" => {
                            let fp = tc.read_file_path()?;
                            match fs::read_to_string(fp.file_path.as_str()) {
                                Ok(res) => res,
                                Err(err) => format!("An error occurred attempting to read file: {err:?}")
                            }
                        }
                        other => {
                            format!("Unsupported tool call: {other}")
                        }
                    };

                    msgs.push(Message {
                        role: "tool".to_string(),
                        content: Some(tool_result),
                        tool_call_id: Some(tc.id.to_string()),
                        tool_calls: None,
                    });
                }

                continue;
            }
            _ => {
                if let Some(content) = ast_msg.content {
                    println!("{content}");
                } else if finish_reason.as_deref() != Some("stop") {
                    eprintln!("Model stopped without content. Reason={finish_reason:?}");
                }

                break;
            }
        }
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
    finish_reason: Option<String>,
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

    #[serde(rename = "type", default = "default_tool_call_type")]
    tool_type: String,

    function: Function,
}

fn default_tool_call_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(unused)]
struct Function {
    name: String,
    arguments: String,
    description: Option<String>,
    parameters: Option<Parameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Parameter {
    #[serde(rename = "type")]
    param_type: String,
    required: Vec<String>,
    properties: Property,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Property {
    #[serde(rename = "file_path")]
    prop_file_path: PropFilePath,
    content: Content,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PropFilePath {
    #[serde(rename = "type")]
    file_path_type: String,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Content {
    #[serde(rename = "type")]
    content_type: String,
    description: String,
}

impl Message {
    fn extract_toolcalls(&self) -> Option<&Vec<ToolCall>> {
        self.tool_calls.as_ref()
    }
}

impl ToolCall {
    fn function_name(&self) -> &str {
        self.function.name.as_str()
    }

    fn arguments(&self) -> &str {
        self.function.arguments.as_str()
    }

    fn read_file_path(&self) -> Result<FilePath, serde_json::Error> {
        serde_json::from_str(self.arguments())
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
