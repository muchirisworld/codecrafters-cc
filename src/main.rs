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

    #[allow(unused_variables)]
    let response: Value = client
        .chat()
        .create_byot(json!({
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt
                }
            ],
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
        }))
        .await?;

    let resp: LLMResponse = serde_json::from_value(response)?;
    let tool_calls: Vec<Option<&str>> = resp.choices.iter().map(|c| {
        extract_filepath(c)
    })
    .collect();
    
    // You can use print statements as follows for debugging, they'll be visible when running tests.
    eprintln!("Logs from your program will appear here!");

    if let Some(content) = &resp.choices[0].message.content {
        println!("{}", content);
    }
    
    if let Some(m) = tool_calls[0] {
        let fp: FilePath = serde_json::from_str(m)?;
        let wr = fs::read_to_string(fp.file_path.as_str())?;
        println!("{wr}");
    }

    Ok(())
}


#[derive(Debug, Serialize, Deserialize)]
struct FilePath {
    file_path: String
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
struct LLMResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
struct Choice {
    index: i32,
    message: Message,
    // finish reason
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
struct Message {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
struct ToolCall {
    id: String,
    index: i32,
    function: Function
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
struct Function {
    name: String,
    arguments: String
}

fn extract_filepath(choice: &Choice) -> Option<&str> {
    if let Some(tool_calls) = &choice.message.tool_calls {
        let fct = &tool_calls[0].function;
        if fct.name == "Read" {
            return Some(fct.arguments.as_str());
        }
    }

    return None;
}
