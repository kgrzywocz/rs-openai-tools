use serde::{Deserialize, Serialize};

/// The role of a message author.
///
/// # Forward compatibility
///
/// OpenAI adds roles over time - `developer` was introduced as the
/// reasoning-model replacement for `system`. Unrecognised roles deserialize
/// into [`Other`](Role::Other) instead of failing the whole response, and the
/// enum is `#[non_exhaustive]` so future additions are not breaking changes.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Role {
    /// System instructions
    #[serde(rename = "system")]
    System,
    /// Developer instructions - the reasoning-model replacement for `system`
    #[serde(rename = "developer")]
    Developer,
    /// End-user input
    #[serde(rename = "user")]
    User,
    /// Model output
    #[serde(rename = "assistant")]
    Assistant,
    /// Legacy function result
    #[serde(rename = "function")]
    Function,
    /// Tool call result
    #[serde(rename = "tool")]
    Tool,
    /// A role this version of the library does not know about
    #[serde(untagged)]
    Other(String),
}

impl TryFrom<String> for Role {
    type Error = &'static str;

    /// Parses a caller-supplied role.
    ///
    /// This stays strict and rejects unknown values, unlike deserialization,
    /// which has to tolerate whatever the API sends. Construct
    /// [`Role::Other`] directly if you need to pass through an unrecognised
    /// role.
    fn try_from(role: String) -> Result<Self, Self::Error> {
        let role = role.to_lowercase();
        match role.as_str() {
            "system" => Ok(Role::System),
            "developer" => Ok(Role::Developer),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "function" => Ok(Role::Function),
            "tool" => Ok(Role::Tool),
            _ => Err("Unknown role"),
        }
    }
}

impl Role {
    /// Returns the wire representation of this role.
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::Developer => "developer",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Function => "function",
            Role::Tool => "tool",
            Role::Other(role) => role.as_str(),
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_role_conversion() {
        assert_eq!(Role::try_from("system".to_string()).unwrap(), Role::System);
        assert_eq!(Role::try_from("developer".to_string()).unwrap(), Role::Developer);
        assert_eq!(Role::try_from("user".to_string()).unwrap(), Role::User);
        assert_eq!(Role::try_from("assistant".to_string()).unwrap(), Role::Assistant);
        assert_eq!(Role::try_from("function".to_string()).unwrap(), Role::Function);
        assert_eq!(Role::try_from("tool".to_string()).unwrap(), Role::Tool);
        assert!(Role::try_from("unknown".to_string()).is_err());
    }

    #[test]
    fn test_role_as_str() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::Developer.as_str(), "developer");
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert_eq!(Role::Function.as_str(), "function");
        assert_eq!(Role::Tool.as_str(), "tool");
        assert_eq!(Role::Other("moderator".to_string()).as_str(), "moderator");
    }
}
