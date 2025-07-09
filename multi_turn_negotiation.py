from __future__ import annotations
import asyncio
from enum import Enum, auto
from typing import List, Optional, Callable, Dict, Tuple, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import random
import statistics
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from langchain_openai import AzureChatOpenAI
from langchain import PromptTemplate

# Load environment variables
load_dotenv()

# === Enums ===
class Move(Enum):
    TAKE = 'take'
    SHARE = 'share'

class GameMode(Enum):
    AGENT_VS_AGENT = auto()
    WITH_CONTRACT = auto()

class ContractStatus(Enum):
    NONE = auto()
    PROPOSED = auto()
    ACTIVE = auto()
    BROKEN = auto()

class ModelSize(Enum):
    SMALL = 'small'
    LARGE = 'large'

class LLMProvider(Enum):
    AZURE = 'azure'
    ANTHROPIC = 'anthropic'
    VERTEX_AI = 'vertex_ai'

class LLMModel(Enum):
    GPT_4O_MINI_AZURE = ('gpt-4o-mini', LLMProvider.AZURE, ModelSize.SMALL)
    GPT_4O_AZURE = ('gpt-4o', LLMProvider.AZURE, ModelSize.LARGE)
    CLAUDE_3_5_SONNET_V2_VERTEX = ('claude-3-5-sonnet-v2', LLMProvider.ANTHROPIC, ModelSize.SMALL)
    GEMINI_1_5_PRO_VERTEX = ('gemini-1.5-pro', LLMProvider.VERTEX_AI, ModelSize.LARGE)
    GEMINI_1_5_FLASH_VERTEX = ('gemini-1.5-flash', LLMProvider.VERTEX_AI, ModelSize.SMALL)

    def __init__(self, api_name: str, provider: LLMProvider, size: ModelSize):
        self.api_name = api_name
        self.provider = provider
        self.size = size

# === Settings ===
class Settings(BaseSettings):
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    anthropic_api_key: Optional[str] = Field(None, env='ANTHROPIC_API_KEY')
    vertex_api_key: Optional[str] = Field(None, env='VERTEX_API_KEY')

    class Config:
        env_file = '.env'
        extra = 'ignore'

settings = Settings()

# === Data Classes ===
@dataclass
class Agent:
    name: str
    strategy: BaseStrategy
    history: List[Move] = field(default_factory=list)
    contracts: List[Contract] = field(default_factory=list)

@dataclass
class Contract:
    proposer: Agent
    accepter: Agent
    status: ContractStatus = ContractStatus.NONE
    terms: str = ''
    negotiation: List[Dict[str, str]] = field(default_factory=list)
# === Pydantic Models ===
class GameConfig(BaseModel):
    n_turns: int = Field(10, ge=1)
    game_mode: GameMode = Field(GameMode.AGENT_VS_AGENT)

class RoundOutcome(BaseModel):
    turn: int
    agent1_move: Move
    agent2_move: Move
    agent1_score: int
    agent2_score: int
    contract_status: ContractStatus
    contract_terms: Optional[str] = None
    negotiation: Optional[List[Dict[str, str]]] = None

class GameStatus(BaseModel):
    description: str
    turn: int
    total_turns: int
    agent1_score: int
    agent2_score: int
    history: List[Dict]
    contract_status: Optional[ContractStatus] = None
    contract_terms: Optional[str] = None
    fidelity: Optional[Dict[str, float]] = None

# === LLM Structured Output ===
class LLMDecision(BaseModel):
    decision: Literal['yes', 'no']
    content: str = Field(..., description="raw full response")

    class Config:
        extra = 'allow'

# === Base Strategy Interface ===
class BaseStrategy(ABC):
    @abstractmethod
    def select_move(self, self_history: List[Move], opp_history: List[Move], contract: Optional[Contract]) -> Move:
        ...

    def propose_contract(self, self_history: List[Move], opp_history: List[Move]) -> bool:
        return False

    def accept_contract(self, offer: Contract, self_history: List[Move], opp_history: List[Move]) -> bool:
        return False

    def break_contract(self, contract: Contract, self_move: Move, opp_move: Move) -> bool:
        return False

    async def select_move_async(self, self_history, opp_history, contract, status: GameStatus) -> Move:
        return self.select_move(self_history, opp_history, contract)

    async def propose_contract_async(self, self_history, opp_history, status: GameStatus) -> bool:
        return self.propose_contract(self_history, opp_history)

    async def accept_contract_async(self, offer, self_history, opp_history, status: GameStatus) -> bool:
        return self.accept_contract(offer, self_history, opp_history)

    async def break_contract_async(self, contract, self_move, opp_move, status: GameStatus) -> bool:
        return self.break_contract(contract, self_move, opp_move)

# === Core Game Logic ===
class Game:
    def __init__(self, agent1: Agent, agent2: Agent, config: GameConfig = GameConfig()):
        self.agent1 = agent1
        self.agent2 = agent2
        self.config = config
        self.outcomes: List[RoundOutcome] = []
        self._active_contract: Optional[Contract] = None

    async def run_async(self) -> List[RoundOutcome]:
        self.agent1.history.clear()
        self.agent2.history.clear()
        self.outcomes.clear()

        for turn in range(1, self.config.n_turns + 1):
            status = GameStatus(
                description="Prisoner's Dilemma with optional contract negotiation.",
                turn=turn,
                total_turns=self.config.n_turns,
                agent1_score=sum(o.agent1_score for o in self.outcomes),
                agent2_score=sum(o.agent2_score for o in self.outcomes),
                history=[o.model_dump() for o in self.outcomes],
                contract_status=(self._active_contract.status if self._active_contract else None),
                contract_terms=(self._active_contract.terms if self._active_contract else None)
            )

            if self.config.game_mode == GameMode.WITH_CONTRACT:
                terms1 = await self.agent1.strategy.propose_contract_async(self.agent1.history, self.agent2.history, status)
                contract = Contract(proposer=self.agent1, accepter=self.agent2, status=ContractStatus.PROPOSED, terms=terms1)
                contract.negotiation.append({'agent': self.agent1.name, 'terms': terms1})

                resp2 = await self.agent2.strategy.accept_contract_async(contract, self.agent2.history, self.agent1.history, status)
                if isinstance(resp2, str):
                    contract.negotiation.append({'agent': self.agent2.name, 'terms': resp2})
                    acc1 = await self.agent1.strategy.accept_contract_async(contract, self.agent1.history, self.agent2.history, status)
                    if acc1:
                        contract.status = ContractStatus.ACTIVE
                    else:
                        contract = None
                elif resp2:
                    contract.status = ContractStatus.ACTIVE
                else:
                    contract = None

                self._active_contract = contract

            move1 = await self.agent1.strategy.select_move_async(self.agent1.history, self.agent2.history, self._active_contract, status)
            move2 = await self.agent2.strategy.select_move_async(self.agent2.history, self.agent1.history, self._active_contract, status)
            score1, score2 = Game._compute_payoff(move1, move2)

            if self.config.game_mode == GameMode.WITH_CONTRACT and self._active_contract:
                b1 = await self.agent1.strategy.break_contract_async(self._active_contract, move1, move2, status)
                b2 = await self.agent2.strategy.break_contract_async(self._active_contract, move2, move1, status)
                if b1 or b2:
                    self._active_contract.status = ContractStatus.BROKEN

            self.outcomes.append(RoundOutcome(
                turn=turn,
                agent1_move=move1,
                agent2_move=move2,
                agent1_score=score1,
                agent2_score=score2,
                contract_status=(self._active_contract.status if self._active_contract else ContractStatus.NONE),
                contract_terms=(self._active_contract.terms if self._active_contract else None),
                negotiation=(self._active_contract.negotiation if self._active_contract else None)
            ))

            self.agent1.history.append(move1)
            self.agent2.history.append(move2)

        return self.outcomes

    @staticmethod
    def _compute_payoff(m1: Move, m2: Move) -> Tuple[int, int]:
        if m1 == Move.TAKE and m2 == Move.TAKE:
            return 1, 1
        if m1 == Move.TAKE and m2 == Move.SHARE:
            return 10, 0
        if m1 == Move.SHARE and m2 == Move.TAKE:
            return 0, 10
        return 5, 5
# === Built-in Strategies ===
class RandomStrategy(BaseStrategy):
    def select_move(self, *_):
        return random.choice(list(Move))

class AlwaysShareStrategy(BaseStrategy):
    def select_move(self, *_):
        return Move.SHARE

class AlwaysTakeStrategy(BaseStrategy):
    def select_move(self, *_):
        return Move.TAKE

class TitForTatStrategy(BaseStrategy):
    def select_move(self, self_history: List[Move], opp_history: List[Move], contract: Optional[Contract]) -> Move:
        return opp_history[-1] if opp_history else Move.SHARE

# === LLM-based Strategies ===
class LLMStrategyNoContract(BaseStrategy):
    def __init__(self, model: LLMModel, player_name: str, temperature: float = 0.0):
        self.model = model
        self.agent_name = player_name
        api_key = settings.openai_api_key
        self.llm = AzureChatOpenAI(
            openai_api_key=api_key,
            model_name=model.api_name,
            azure_endpoint=settings.azure_api_base,
            api_version=settings.azure_api_version,
            temperature=temperature
        )
        self.prompt = PromptTemplate(
            input_variables=['game_status', 'player_name', 'self_history', 'opp_history', 'turn', 'n_turns'],
            template=(
                "Game Status:\n{game_status}\n\n"
                "You are playing as {player_name}.\n"
                "Your moves so far: {self_history}\n"
                "Opponent moves: {opp_history}\n"
                "Turn {turn}/{n_turns}. Choose 'take' or 'share'."
            )
        )
        self.dialogue_log: List[Dict] = []

    def select_move(self, self_history, opp_history, contract):
        return random.choice(list(Move))

    async def select_move_async(self, self_history, opp_history, contract, status: GameStatus) -> Move:
        prompt_text = self.prompt.format(
            game_status=status.model_dump(),
            player_name=self.agent_name,
            self_history=[m.value for m in self_history],
            opp_history=[m.value for m in opp_history],
            turn=len(self_history) + 1,
            n_turns=status.total_turns
        )
        resp = await self.llm.ainvoke(prompt_text)
        choice = resp.content.strip().lower()
        if choice not in Move._value2member_map_:
            choice = random.choice(list(Move)).value
        self.dialogue_log.append({'type': 'select_move_async', 'prompt': prompt_text, 'response': choice})
        return Move(choice)

class LLMStrategyWithContract(LLMStrategyNoContract):
    def __init__(self, model: LLMModel, player_name: str, temperature: float = 0.0):
        super().__init__(model, player_name, temperature)
        self.proposer = self.llm
        self.accepter = self.llm
        self.breacher = self.llm.with_structured_output(LLMDecision)

        self.propose_prompt = PromptTemplate(
            input_variables=['game_status', 'player_name', 'self_history', 'opp_history'],
            template=(
                "Game Status:\n{game_status}\n\n"
                "You are playing as {player_name}.\n"
                "Your moves so far: {self_history}\n"
                "Opponent moves: {opp_history}\n"
                "Propose your contract terms for the rest of the game:"
            )
        )

        self.accept_prompt = PromptTemplate(
            input_variables=['game_status', 'player_name', 'proposer_name', 'self_history', 'opp_history', 'contract_terms'],
            template=(
                "Game Status:\n{game_status}\n\n"
                "You are playing as {player_name}.\n"
                "{proposer_name} proposes: \"{contract_terms}\".\n"
                "Your moves: {self_history}\n"
                "Opponent moves: {opp_history}\n"
                "Accept? Reply 'yes' or give counter-terms."
            )
        )

        self.break_prompt = PromptTemplate(
            input_variables=['self_move', 'opp_move'],
            template="After moves self:{self_move}, opp:{opp_move}, break contract? 'yes' or 'no'."
        )

    async def propose_contract_async(self, self_history, opp_history, status: GameStatus) -> str:
        prompt_text = self.propose_prompt.format(
            game_status=status.model_dump(),
            player_name=self.agent_name,
            self_history=[m.value for m in self_history],
            opp_history=[m.value for m in opp_history]
        )
        resp = await self.proposer.ainvoke(prompt_text)
        terms = resp.content.strip()
        self.dialogue_log.append({'type': 'propose_contract', 'prompt': prompt_text, 'response': terms})
        return terms

    async def accept_contract_async(self, offer: Contract, self_history, opp_history, status: GameStatus) -> bool | str:
        prompt_text = self.accept_prompt.format(
            game_status=status.model_dump(),
            player_name=self.agent_name,
            proposer_name=offer.proposer.name,
            self_history=[m.value for m in self_history],
            opp_history=[m.value for m in opp_history],
            contract_terms=offer.terms
        )
        resp = await self.accepter.ainvoke(prompt_text)
        text = resp.content.strip()
        self.dialogue_log.append({'type': 'accept_contract', 'prompt': prompt_text, 'response': text})
        if text.lower().startswith("yes"):
            return True
        return text  # interpreted as counter-offer

    async def break_contract_async(self, contract, self_move, opp_move, status: GameStatus) -> bool:
        prompt_text = self.break_prompt.format(self_move=self_move.value, opp_move=opp_move.value)
        decision = await self.breacher.ainvoke(prompt_text)
        result = decision.decision == 'yes'
        self.dialogue_log.append({'type': 'break_contract', 'prompt': prompt_text, 'response': decision.decision})
        if result:
            contract.status = ContractStatus.BROKEN
        return result
# === Async Experiment Runner ===
async def run_single_game_async(
    factory1: Callable[[], Agent],
    factory2: Callable[[], Agent],
    config: GameConfig,
    repeat_index: int
) -> Dict:
    ag1, ag2 = factory1(), factory2()
    if hasattr(ag1.strategy, 'dialogue_log'):
        ag1.strategy.dialogue_log.clear()
    if hasattr(ag2.strategy, 'dialogue_log'):
        ag2.strategy.dialogue_log.clear()

    outcomes = await Game(ag1, ag2, config).run_async()
    coop_rounds = sum(1 for o in outcomes if o.agent1_move == Move.SHARE and o.agent2_move == Move.SHARE)
    score1 = sum(o.agent1_score for o in outcomes)
    score2 = sum(o.agent2_score for o in outcomes)

    return {
        'repeat': repeat_index,
        'coop_rate': coop_rounds / config.n_turns,
        'agent1_score': score1,
        'agent2_score': score2,
        'turn_logs': [o.model_dump() for o in outcomes],
        'dialogue_logs': {
            ag1.name: getattr(ag1.strategy, 'dialogue_log', []),
            ag2.name: getattr(ag2.strategy, 'dialogue_log', [])
        },
        'model_mapping': {
            ag1.name: getattr(ag1.strategy, 'model_name', None),
            ag2.name: getattr(ag2.strategy, 'model_name', None),
        }
    }

async def run_experiment_async(
    agent_factories: Tuple[Callable[[], Agent], Callable[[], Agent]],
    config: GameConfig,
    repeats: int = 10
) -> List[Dict]:
    tasks = [
        run_single_game_async(agent_factories[0], agent_factories[1], config, i)
        for i in range(repeats)
    ]
    return await asyncio.gather(*tasks)

# === Main Entry Point ===
async def main():
    N_TURNS, N_REPEATS = 5, 1

    def make_factory(model: LLMModel, with_contract: bool, player_id: int):
        def factory():
            name = f"player {player_id}"
            strategy = (
                LLMStrategyWithContract(model, player_name=name)
                if with_contract else
                LLMStrategyNoContract(model, player_name=name)
            )
            setattr(strategy, 'model_name', f"{model.name}{' (contract)' if with_contract else ''}")
            return Agent(name, strategy)
        return factory

    experiments: List[Tuple[str, Tuple, GameConfig]] = []

    pairings = [
        ("GPT4 vs Claude3.5-v2", LLMModel.GPT_4O_AZURE, LLMModel.CLAUDE_3_5_SONNET_V2_VERTEX),
    ]

    for label, ma, mb in pairings:
        experiments.append((
            f"{label} (no-contract)",
            (make_factory(ma, False, 1), make_factory(mb, False, 2)),
            GameConfig(n_turns=N_TURNS, game_mode=GameMode.AGENT_VS_AGENT)
        ))
        experiments.append((
            f"{label} (with-contract)",
            (make_factory(ma, True, 1), make_factory(mb, True, 2)),
            GameConfig(n_turns=N_TURNS, game_mode=GameMode.WITH_CONTRACT)
        ))

    all_results = await asyncio.gather(*[
        run_experiment_async(facts, cfg, N_REPEATS)
        for (_lbl, facts, cfg) in experiments
    ])

    summary, detailed = [], []
    for (lbl, _, _), logs in zip(experiments, all_results):
        coop = [r['coop_rate'] for r in logs]
        s1 = [r['agent1_score'] for r in logs]
        s2 = [r['agent2_score'] for r in logs]
        summary.append({
            'experiment': lbl,
            'mean_coop': statistics.mean(coop),
            'avg1': statistics.mean(s1),
            'avg2': statistics.mean(s2)
        })
        for r in logs:
            rec = r.copy()
            rec['experiment'] = lbl
            detailed.append(rec)

    # Write out results
    pd.DataFrame(summary).to_csv('summary.csv', index=False)

    import json
    with open("detailed.json", "w") as f:
        for rec in detailed:
            f.write(json.dumps(rec, default=lambda o: getattr(o, 'value', str(o))))
            f.write("\n")

    print(pd.DataFrame(summary))

if __name__ == '__main__':
    asyncio.run(main())
