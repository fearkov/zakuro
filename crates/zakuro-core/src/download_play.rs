//! Download Play as a guest, what the console's own Download Play does
//! hosts' offers in the room, joining one, getting the child when the host sends it,
//! and the passphrase of the network the host puts up next
//! the child then runs w/ that, see Config::wireless_reboot

use std::time::{Duration, Instant};

use crate::room::{Link, Mac};
use crate::services::dlp::Offer;
use crate::services::uds::Wireless;

/// an offer heard from a host, and when last
#[derive(Debug, Clone)]
pub struct Heard {
    pub host: Mac,
    pub offer: Offer,
    pub at: Instant,
}

/// where the guest is at
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// asked to join, no answer yet
    Asking,
    /// let in, waiting for the host's game to start sending
    Waiting,
    /// getting the child, this much of it
    Receiving { got: u64, total: u64 },
    /// has the child, waiting for the game to start
    Ready,
    /// game starts, the child (none if the guest had it) and what it runs w/
    Begun { child: Option<Vec<u8>>, reboot: [u8; 16] },
    /// went wrong, and why
    Failed(String),
}

struct Session {
    host: Mac,
    /// guest already has the child, nothing to send
    has_child: bool,
    received: Vec<u8>,
    phase: Phase,
}

/// a Download Play guest in a room
pub struct Guest {
    link: Link,
    heard: Vec<Heard>,
    session: Option<Session>,
}

/// an offer not heard again for this long is gone
const FORGET: Duration = Duration::from_secs(4);

/// the node a guest joins as, what the host's game shows, console number and player name
pub fn node(console_id: u64, nickname: &str) -> Vec<u8> {
    let mut node = vec![0u8; 0x28];
    node[..8].copy_from_slice(&console_id.to_le_bytes());
    for (i, unit) in nickname.encode_utf16().take(10).enumerate() {
        node[8 + i * 2..10 + i * 2].copy_from_slice(&unit.to_le_bytes());
    }
    node
}

impl Guest {
    pub fn new(link: Link) -> Guest {
        Guest { link, heard: Vec::new(), session: None }
    }

    /// offers in the room, first heard first
    pub fn offers(&self) -> &[Heard] {
        &self.heard
    }

    pub fn phase(&self) -> Option<&Phase> {
        self.session.as_ref().map(|session| &session.phase)
    }

    /// asks the host to join as node, has_child means the child's already at hand
    pub fn join(&mut self, host: Mac, node: Vec<u8>, has_child: bool) {
        self.leave();
        self.send(host, &Wireless::DlpJoin { node });
        self.session = Some(Session { host, has_child, received: Vec::new(), phase: Phase::Asking });
    }

    /// ends the session once it's over (game started or failed) and says how it ended
    pub fn end(&mut self) -> Option<Phase> {
        let over = matches!(self.phase(), Some(Phase::Begun { .. } | Phase::Failed(_)));
        self.session.take_if(|_| over).map(|session| session.phase)
    }

    pub fn leave(&mut self) {
        if let Some(session) = self.session.take() {
            if !matches!(session.phase, Phase::Begun { .. } | Phase::Failed(_)) {
                self.send(session.host, &Wireless::DlpLeave);
            }
        }
    }

    fn send(&self, to: Mac, message: &Wireless) {
        if let Ok(body) = postcard::to_allocvec(message) {
            self.link.send(to, body);
        }
    }

    /// takes in what the room brought
    pub fn poll(&mut self) {
        for (from, body) in self.link.receive() {
            let Ok(message) = postcard::from_bytes::<Wireless>(&body) else { continue };
            self.take(from, message);
        }
        self.heard.retain(|heard| heard.at.elapsed() < FORGET);
        if !self.link.open() {
            if let Some(session) = &mut self.session {
                if !matches!(session.phase, Phase::Begun { .. }) {
                    session.phase = Phase::Failed("the room closed".into());
                }
            }
        }
    }

    fn take(&mut self, from: Mac, message: Wireless) {
        if let Wireless::DlpOffer(offer) = message {
            let at = Instant::now();
            match self.heard.iter_mut().find(|heard| heard.host == from) {
                Some(heard) => *heard = Heard { host: from, offer, at },
                None => self.heard.push(Heard { host: from, offer, at }),
            }
            return;
        }
        let Some(session) = self.session.as_mut().filter(|session| session.host == from) else { return };
        let mut reply = None;
        match message {
            Wireless::DlpTaken { .. } if session.phase == Phase::Asking => session.phase = Phase::Waiting,
            Wireless::DlpRefused => session.phase = Phase::Failed("the host turned this console away".into()),
            Wireless::DlpLeave => session.phase = Phase::Failed("the host ended the session".into()),
            Wireless::DlpStart => {
                let needs = !session.has_child;
                session.phase = if needs { Phase::Receiving { got: 0, total: 0 } } else { Phase::Ready };
                reply = Some(Wireless::DlpNeed { needs });
            }
            Wireless::DlpPiece { offset, total, data } => {
                if let Phase::Receiving { .. } = session.phase {
                    if session.received.len() as u64 == offset && total <= 64 << 20 {
                        session.received.extend_from_slice(&data);
                        let got = session.received.len() as u64;
                        session.phase = if got >= total { Phase::Ready } else { Phase::Receiving { got, total } };
                        if got >= total {
                            reply = Some(Wireless::DlpReady);
                        }
                    }
                }
            }
            Wireless::DlpBegin { passphrase } if session.phase == Phase::Ready => {
                let child = (!session.has_child).then(|| std::mem::take(&mut session.received));
                let mut reboot = [0u8; 16];
                reboot[..6].copy_from_slice(&from);
                let length = passphrase.len().min(9);
                reboot[6..6 + length].copy_from_slice(&passphrase[..length]);
                session.phase = Phase::Begun { child, reboot };
            }
            _ => {}
        }
        if let Some(reply) = reply {
            self.send(from, &reply);
        }
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        self.leave();
    }
}
