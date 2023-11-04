use rust_bert::pipelines::summarization::SummarizationModel;
use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel, Answer};
use std::fs;
use clap::{Parser, Subcommand};
use std::convert::TryFrom;

#[derive(Parser)]
#[clap(about, version, author)]
struct PromptOptions {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    French {},
    Spanish {},
    Italian {},
    Summarize {},
    Question {},
}

fn main() {

    let value = PromptOptions::parse();

    match &value.command {

        Commands::French {} => {
            let response = french_translate();
            println!("{:?}", &response);
        },
        Commands::Spanish {} => {
            let response = spanish_translate();
            println!("{:?}", &response);
        },
        Commands::Italian {} => {
            let response = italian_translate();
            println!("{:?}", &response);
        },
        Commands::Summarize {} => {
            let response = summarize();
            println!("{:?}", &response);
        },
        Commands::Question {} => {
            let response = questioning();
            println!("{:?}", &response);
        }
    }
}

fn questioning() -> anyhow::Result<()> {
    let model = QuestionAnsweringModel::new(Default::default())?;

    // let mut input= vec![];

    let story = fs::read_to_string("story.txt")
        .expect("Should be able to read file!");

    // input.push(story);

    let question = String::from("What country is this story about?");
    let context = story;

    let answers = model.predict(&[QaInput { question, context }], 1, 32);

    let pos = answers[0].get(0).unwrap();

    println!("{}", pos.answer);

   Ok(()) 
}

fn summarize() -> anyhow::Result<()> {
    let model = SummarizationModel::new(Default::default())?;

    let mut input= vec![];

    let story = fs::read_to_string("story.txt")
        .expect("Should be able to read file!");

    input.push(story);

    let output = model.summarize(&input);

    println!("{:?}", output);

    Ok(())
}


fn italian_translate() -> anyhow::Result<()> {

    let mut input= vec![];

    let story = fs::read_to_string("story.txt")
        .expect("Should be able to read file!");

    input.push(story);
    
    let model = TranslationModelBuilder::new()
        .with_source_languages(vec![Language::English])
        .with_target_languages(vec![Language::Spanish, Language::French, Language::Italian])
        .create_model()?;

    let output = model.translate(&input, None, Language::Italian)?;
    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())

    }

fn spanish_translate() -> anyhow::Result<()> {

    let mut input= vec![];

    let story = fs::read_to_string("story.txt")
        .expect("Should be able to read file!");

    input.push(story);
    
    let model = TranslationModelBuilder::new()
        .with_source_languages(vec![Language::English])
        .with_target_languages(vec![Language::Spanish, Language::French, Language::Italian])
        .create_model()?;

    let output = model.translate(&input, None, Language::Spanish)?;
    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())

    }

fn french_translate() -> anyhow::Result<()> {

    let mut input= vec![];

    let story = fs::read_to_string("story.txt")
        .expect("Should be able to read file!");

    input.push(story);
    
    let model = TranslationModelBuilder::new()
        .with_source_languages(vec![Language::English])
        .with_target_languages(vec![Language::Spanish, Language::French, Language::Italian])
        .create_model()?;

    let output = model.translate(&input, None, Language::French)?;
    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())

    }
