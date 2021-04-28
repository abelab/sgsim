import asyncio
import sys
import traceback
from abc import ABC, abstractmethod

from colorama import Fore, Style

from event_sim import eventsim

NETWORK_LATENCY = 1
verbose_output = True


class Event(ABC):
    def __init__(self, receiver: 'LocalNode' = None):
        # self.origin: LocalNode or None = None
        self.sender: 'LocalNode' or None = None
        self.receiver = receiver
        self.send_time = None
        self.vtime = None

    @abstractmethod
    async def run(self, node: 'LocalNode'):
        pass


class SchedEvent(Event):
    def __init__(self, job):
        self.job = job
        self.is_canceled = False
        super().__init__()

    def __str__(self):
        return "SchedEvent(receiver={}{})".format(self.receiver, ' (canceled)' if self.is_canceled else '')

    async def run(self, node: 'LocalNode'):
        if not self.is_canceled:
            await self.job()

    def cancel(self):
        self.is_canceled = True

    def plot_string(self):
        pass


class RequestEvent(Event):
    nextEventId = 1000
    nextAckId = 10000

    def __init__(self, receiver: 'LocalNode'):
        self.eventId = RequestEvent.nextEventId
        RequestEvent.nextEventId += 1
        self.ackId = None
        self.isRequestingNode = True
        # self.onReply = None
        self.future = asyncio.get_event_loop().create_future()
        super().__init__(receiver)

    async def getReply(self):
        return await self.future


class ReplyEvent(Event):
    def __init__(self, req: RequestEvent):
        self.req = req
        super().__init__(req.sender)

    async def run(self, node: 'LocalNode'):
        # EventExecutor.unregisterRequestMessage(self.req.eventId)
        # self.req.onReply(self)
        self.req.future.set_result(self)


class EventExecutor:
    class FinishException(Exception):
        def __init__(self):
            super().__init__()

    log: list[Event] = []
    loop = None

    @classmethod
    def reset(cls):
        EventExecutor.log = []
        EventExecutor.loop = eventsim.EventSimulator()
        asyncio.set_event_loop(EventExecutor.loop)

    @classmethod
    def vtime(cls) -> float:
        return EventExecutor.loop.time()

    @classmethod
    def register_event(cls, ev: Event, latency=NETWORK_LATENCY):
        ev.send_time = EventExecutor.vtime()
        ev.vtime = EventExecutor.vtime() + latency

        async def invoke():
            log(f"*** time={EventExecutor.loop.time()}, {ev}")
            await ev.run(ev.receiver)

        EventExecutor.log.append(ev)
        EventExecutor.loop.call_at(ev.vtime, lambda: asyncio.ensure_future(invoke()))

    @classmethod
    def sim(cls, simulation_time: int, verbose=True):
        async def stop():
            await asyncio.sleep(simulation_time, loop=EventExecutor.loop)
            raise EventExecutor.FinishException()

        print("EventExecutor.sim(): started")
        global verbose_output
        verbose_output = verbose

        # init
        asyncio.ensure_future(stop())
        try:
            EventExecutor.loop.run_forever()
        except EventExecutor.FinishException:
            # log("EventExecutor.sim(): finished")
            pass
        except Exception:
            print(traceback.format_exc())
            sys.exit(1)

        # loop.run_forever()
        # while len(EventExecutor.queue) > 0 and EventExecutor.vtime < max_vtime:
        #     (vtime, count, event) = heapq.heappop(EventExecutor.queue)
        #     EventExecutor.vtime = vtime
        #     if isinstance(event, SchedEvent) and event.is_canceled:
        #         pass
        #     else:
        #         print("*** T{}: {}: {}".format(vtime, event.receiver, event))
        #     event.run(event.receiver)
        #     EventExecutor.n_event += 1
        #     EventExecutor.log.append(event)
        print("EventExecutor.sim(): finished")

    @classmethod
    async def sleep(cls, delay):
        await asyncio.sleep(delay, loop=EventExecutor.loop)


class AbstractNode:
    def send_event(self, ev: Event):
        ev.sender = self
        EventExecutor.register_event(ev)

    def sched(self, delay, job) -> SchedEvent:
        ev = SchedEvent(job)
        EventExecutor.register_event(ev, latency=delay)
        return ev


def log(*args):
    if verbose_output:
        print(Fore.CYAN, end="")
        print(*args)
        print(Style.RESET_ALL, end="")


EventExecutor.reset()

