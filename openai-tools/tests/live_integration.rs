//! Integration tests for the OpenAI Live API.
//!
//! These tests require `OPENAI_API_KEY`, connect to the real API, and may incur
//! usage charges.
//!
//! Run with:
//! ```bash
//! cargo nextest run --test live_integration
//! ```

use std::time::Duration;

use openai_tools::live::{CloseReason, LiveClient, ServerEvent};

#[tokio::test]
async fn test_live_session_start_and_close() {
    let session = LiveClient::new().connect().await.expect("Live session should start");
    assert!(session.session_id().starts_with("live_"));

    let closed = session.close().await.expect("Live session should close").expect("server should send session.closed");
    assert_eq!(closed.reason, CloseReason::CloseRequested);
}

#[tokio::test]
async fn test_live_commentary_produces_speech() {
    let mut client = LiveClient::new();
    client.instructions("Greet the user briefly as soon as the session starts.");

    let session = client.connect().await.expect("Live session should start");
    let (sender, mut receiver) = session.split();
    let audio_sender = sender.clone();

    let audio_task = tokio::spawn(async move {
        let silence = vec![0_u8; 4_800];
        let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
        while tokio::time::Instant::now() < deadline {
            audio_sender.append_audio_bytes(&silence).await?;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    });

    tokio::time::sleep(Duration::from_secs(2)).await;
    sender.append_commentary("Tell the user: Hello from the Live API test.", None).await.expect("commentary should be sent");

    let mut got_audio = false;
    let mut got_transcript = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);

    while tokio::time::Instant::now() < deadline && !(got_audio && got_transcript) {
        let event = match tokio::time::timeout(Duration::from_secs(2), receiver.recv()).await {
            Ok(result) => result.expect("Live event should deserialize").expect("WebSocket should stay open"),
            Err(_) => continue,
        };

        match event {
            ServerEvent::OutputAudioDelta(_) => got_audio = true,
            ServerEvent::OutputTranscriptDelta(_) => got_transcript = true,
            ServerEvent::Error(event) => panic!("Live API error: {}", event.error.message),
            _ => {}
        }
    }

    audio_task.abort();
    let _ = audio_task.await;
    receiver.close().await.expect("Live session should close");

    assert!(got_audio, "should receive at least one output audio delta");
    assert!(got_transcript, "should receive at least one output transcript delta");
}
