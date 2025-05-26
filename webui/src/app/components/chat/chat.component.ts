/**
 * @license
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {HttpErrorResponse} from '@angular/common/http';
import {AfterViewChecked, AfterViewInit, Component, ElementRef, inject, OnDestroy, OnInit, signal, ViewChild, WritableSignal,} from '@angular/core';
// FormControl removed as selectedAppControl is removed
import {MatDialog} from '@angular/material/dialog'; // MatDialogModule already imported via ComponentModule
import {MatPaginatorIntl} from '@angular/material/paginator';
import {MatDrawer} from '@angular/material/sidenav';
import {MatSnackBar} from '@angular/material/snack-bar';
import {DomSanitizer, SafeHtml} from '@angular/platform-browser';
// ActivatedRoute, NavigationEnd, Router removed as URL sync for app name is removed
import {instance} from '@viz-js/viz';
import {catchError, Observable, of, shareReplay, switchMap, take, tap, filter, map, distinctUntilChanged } from 'rxjs'; // filter, map, distinctUntilChanged might not be needed if URL sync is gone

import {URLUtil} from '../../../utils/url-util';
import {AgentRunRequest} from '../../core/models/AgentRunRequest';
import {Session} from '../../core/models/Session';
import {AgentService} from '../../core/services/agent.service';
import {ArtifactService} from '../../core/services/artifact.service';
import {AudioService} from '../../core/services/audio.service';
import {EventService} from '../../core/services/event.service';
import {SessionService} from '../../core/services/session.service';
import {VideoService} from '../../core/services/video.service';
import {WebSocketService} from '../../core/services/websocket.service';
// ResizableDrawerDirective removed from imports as it's a directive, not typically imported here. It's declared in ComponentModule.
import {isArtifactAudio, isArtifactImage, openBase64InNewTab} from '../artifact-tab/artifact-tab.component';
// AudioPlayerComponent removed from imports, it's declared in ComponentModule.
import {EvalTabComponent} from '../eval-tab/eval-tab.component';
import {EventTabComponent} from '../event-tab/event-tab.component';
import {PendingEventDialogComponent} from '../pending-event-dialog/pending-event-dialog.component';
import {DeleteSessionDialogComponent, DeleteSessionDialogData,} from '../session-tab/delete-session-dialog/delete-session-dialog.component';
import {SessionTabComponent} from '../session-tab/session-tab.component';
import {ViewImageDialogComponent} from '../view-image-dialog/view-image-dialog.component';
import { ActivatedRoute } from '@angular/router'; // Keep for other queryParams if needed, or remove if not.

// Define the hardcoded agent name here
const HARDCODED_AGENT_NAME = 'expert-agents'; 

function fixBase64String(base64: string): string {
  // Replace URL-safe characters if they exist
  base64 = base64.replace(/-/g, '+').replace(/_/g, '/');

  // Fix base64 padding
  while (base64.length % 4 !== 0) {
    base64 += '=';
  }

  return base64;
}

class CustomPaginatorIntl extends MatPaginatorIntl {
  override nextPageLabel = 'Next Event';
  override previousPageLabel = 'Previous Event';
  override firstPageLabel = 'First Event';
  override lastPageLabel = 'Last Event';

  override getRangeLabel = (page: number, pageSize: number, length: number) => {
    if (length === 0) {
      return `Event 0 of ${length}`;
    }

    length = Math.max(length, 0);
    const startIndex = page * pageSize;

    return `Event ${startIndex + 1} of ${length}`;
  };
}

const BIDI_STREAMING_RESTART_WARNING =
    'Restarting bidirectional streaming is not currently supported. Please refresh the page or start a new session.';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
  standalone: false,
  providers: [{provide: MatPaginatorIntl, useClass: CustomPaginatorIntl}],
})
export class ChatComponent implements OnInit, AfterViewInit, OnDestroy,
                                      AfterViewChecked {
  @ViewChild('videoContainer', {read: ElementRef}) videoContainer!: ElementRef;
  @ViewChild('sideDrawer') sideDrawer!: MatDrawer;
  @ViewChild(EventTabComponent) eventTabComponent!: EventTabComponent;
  @ViewChild(SessionTabComponent) sessionTab!: SessionTabComponent;
  @ViewChild(EvalTabComponent) evalTab!: EvalTabComponent;
  @ViewChild('autoScroll') private scrollContainer!: ElementRef;
  private _snackBar = inject(MatSnackBar);
  shouldShowEvalTab = signal(true);
  enableSseIndicator = signal(false);
  videoElement!: HTMLVideoElement;
  currentMessage = '';
  messages: any[] = [];
  lastTextChunk: string = '';
  streamingTextMessage: any|null = null;
  artifacts: any[] = [];
  userInput: string = '';
  userId = 'user';
  appName = ''; // Will be set to HARDCODED_AGENT_NAME
  sessionId = ``;
  isAudioRecording = false;
  isVideoRecording = false;
  longRunningEvents: any[] = [];
  functionCallEventId = '';
  redirectUri = URLUtil.getBaseUrlWithoutPath();
  showSidePanel = true;
  useSse = false;
  currentSessionState = {};

  // Make hardcoded agent name available to the template
  public readonly templateHardcodedAgentName = HARDCODED_AGENT_NAME;

  // TODO: Remove this once backend supports restarting bidi streaming.
  sessionHasUsedBidi = new Set<string>();

  eventData = new Map<string, any>();
  traceData: any[] = [];
  eventMessageIndexArray: any[] = [];
  renderedEventGraph: SafeHtml|undefined;
  rawSvgString: string|null = null;

  selectedEvent: any = undefined;
  selectedEventIndex: any = undefined;
  llmRequest: any = undefined;
  llmResponse: any = undefined;
  llmRequestKey = 'gcp.vertex.agent.llm_request';
  llmResponseKey = 'gcp.vertex.agent.llm_response';

  selectedFiles: {file: File; url: string}[] = [];
  private previousMessageCount = 0;

  protected isArtifactImage = isArtifactImage;
  protected openBase64InNewTab = openBase64InNewTab;
  protected isArtifactAudio = isArtifactAudio;

  // Removed: selectedAppControl
  // Removed: apps$ observable for dropdown

  // State for loading the hardcoded agent
  protected isLoadingApps: WritableSignal<boolean> = signal(false); // Renamed for clarity, effectively "isLoadingAgent"
  protected loadingError: WritableSignal<string> = signal('');

  // Flag for starter questions
  showStarterQuestions = false;

  // Removed: router and activatedRoute for app name sync. Kept route for other potential uses.
  private readonly agentService = inject(AgentService);

  constructor(
      private sanitizer: DomSanitizer,
      private sessionService: SessionService,
      private artifactService: ArtifactService,
      private audioService: AudioService,
      private webSocketService: WebSocketService,
      private videoService: VideoService,
      private dialog: MatDialog,
      private eventService: EventService,
      private route: ActivatedRoute, // Keep if other query params are used, otherwise remove
  ) {}

  ngOnInit(): void {
    this.isLoadingApps.set(true);
    this.appName = ''; // Ensure appName is initially empty
    this.showStarterQuestions = false; // Ensure it's false during loading

    // Validate and select the hardcoded agent
    this.agentService.listApps().pipe(
      catchError((err: HttpErrorResponse) => {
        this.loadingError.set(`Failed to connect to the ADK API server to list available agents. Please ensure it's running. Error: ${err.message}`);
        this.isLoadingApps.set(false);
        return of(undefined);
      })
    ).subscribe(availableApps => {
      this.isLoadingApps.set(false);
      if (availableApps && availableApps.includes(HARDCODED_AGENT_NAME)) {
        this.selectApp(HARDCODED_AGENT_NAME); // This will set this.appName and call createSession
        this.loadingError.set('');
      } else if (availableApps) {
        const availableAgentsString = availableApps.length > 0 ? `Available agents: [${availableApps.join(', ')}]` : "No agents are available on the server.";
        this.loadingError.set(`Error: The hardcoded agent "${HARDCODED_AGENT_NAME}" was not found on the server. ${availableAgentsString}. Please check the agent name or the server configuration.`);
      } else if (!this.loadingError()) {
        this.loadingError.set(`An unexpected issue occurred while trying to load agent "${HARDCODED_AGENT_NAME}".`);
      }
    });

    this.webSocketService.onCloseReason().subscribe((closeReason) => {
      const error =
          'Please check server log for full details: \n' + closeReason;
      this.openSnackBar(error, 'OK');
    });

    const location = new URL(window.location.href);
    const searchParams = location.searchParams;
    if (searchParams.has('code')) {
      const authResponseUrl = window.location.href;
      window.opener?.postMessage({authResponseUrl}, window.origin);
      window.close();
    }

    this.agentService.getLoadingState().subscribe((isLoading: boolean) => {
      if (isLoading) {
        if (!(this.messages.length > 0 && this.messages[this.messages.length - 1].isLoading)) {
           this.messages.push({role: 'bot', isLoading: true});
        }
      } else if (this.messages[this.messages.length - 1] &&
        this.messages[this.messages.length - 1].isLoading) {
        this.messages.pop();
      }
    });
  }

  ngAfterViewInit() {
    this.showSidePanel = true;
    this.sideDrawer.open();
  }

  ngAfterViewChecked() {
    if (this.messages.length !== this.previousMessageCount) {
      this.scrollContainer.nativeElement.scrollTop =
          this.scrollContainer.nativeElement.scrollHeight;
      this.previousMessageCount = this.messages.length;
    }
  }

  selectApp(hardcodedAppName: string) {
    if (this.appName === hardcodedAppName && this.sessionId) {
      return;
    }

    this.appName = hardcodedAppName;
    this.agentService.setApp(hardcodedAppName);

    this.eventData = new Map<string, any>();
    this.traceData = [];
    this.eventMessageIndexArray = [];
    this.messages = [];
    this.artifacts = [];
    this.userInput = '';
    this.longRunningEvents = [];
    this.currentSessionState = {};
    this.sessionId = '';
    this.showStarterQuestions = false; // Reset during app selection phase

    if (this.isAudioRecording) {
        this.stopAudioRecording();
        this.isAudioRecording = false;
    }
    if (this.isVideoRecording) {
        this.stopVideoRecording();
        this.isVideoRecording = false;
    }

    this.createSession();
  }

  createSession() {
    if (!this.appName) {
      console.error("Cannot create session: appName is not set.");
      this.loadingError.set(`Internal error: Agent name ("${HARDCODED_AGENT_NAME}") not available for session creation.`);
      this.showStarterQuestions = false;
      return;
    }
    this.showStarterQuestions = false; // Ensure false before session creation completes

    this.sessionService.createSession(this.userId, this.appName)
        .pipe(catchError((err: HttpErrorResponse) => {
          this.loadingError.set(`Failed to create session for agent "${this.appName}". Error: ${err.message}`);
          this.sessionId = '';
          this.showStarterQuestions = false;
          return of({id: '', state: {}} as Session);
        }))
        .subscribe((res: Session) => {
          if (res && res.id) {
            this.currentSessionState = res.state;
            this.sessionId = res.id;
            this.messages = []; // Crucial: clear messages for a new session
            this.showStarterQuestions = true; // Show starters for a new, empty session
            if (this.sessionTab) {
              this.sessionTab.refreshSession();
            }
            if (this.loadingError() && this.loadingError().startsWith("Failed to create session")) {
            } else if (!this.loadingError().includes("not found on the server")) {
              this.loadingError.set('');
            }
          } else if (!this.loadingError()) {
            this.loadingError.set(`Session creation for agent "${this.appName}" did not return a valid session ID.`);
            this.sessionId = '';
            this.showStarterQuestions = false;
          }
        });
  }

  async sendMessage(event: Event) {
    event.preventDefault();
    if (!this.userInput.trim()) return;

    this.showStarterQuestions = false; // Hide starters as soon as a message is being sent

    if (this.selectedFiles.length > 0) {
      const messageAttachments = this.selectedFiles.map((file) => ({
                                                          file: file.file,
                                                          url: file.url,
                                                        }));
      this.messages.push({role: 'user', attachments: messageAttachments});
    }
    this.messages.push({role: 'user', text: this.userInput});

    const req: AgentRunRequest = {
      appName: this.appName,
      userId: this.userId,
      sessionId: this.sessionId,
      newMessage: {
        'role': 'user',
        'parts': await this.getUserMessageParts(),
      },
      streaming: this.useSse,
    };
    this.selectedFiles = [];
    let index = this.eventMessageIndexArray.length - 1;
    this.streamingTextMessage = null;
    this.agentService.runSse(req).subscribe({
      next: async (chunk) => {
        this.agentService.getLoadingState().next(false);
        if (chunk.startsWith('{"error"')) {
          this.openSnackBar(chunk, 'OK');
          return;
        }
        const chunkJson = JSON.parse(chunk);
        if (chunkJson.error) {
          this.openSnackBar(chunkJson.error, 'OK');
          return;
        }
        if (chunkJson.content) {
          for (let part of chunkJson.content.parts) {
            index += 1;
            this.processPart(chunkJson, part, index);
          }
          if (chunkJson.actions) {
            if (chunkJson.actions.skip_summarization || chunkJson.actions.skipSummarization) {
              if (chunkJson.content.parts) {
                const adkPart = chunkJson.content.parts[0];
                if (adkPart.functionResponse && adkPart.functionResponse.response && adkPart.functionResponse.response.result && adkPart.functionResponse.response.result.parts.length > 0) {
                  const resultPart = adkPart.functionResponse.response.result.parts[0];
                  if (resultPart.text) {
                    this.streamingTextMessage = {
                      role: 'bot',
                      text: this.processThoughtText(resultPart.text),
                      thought: false,
                      eventId: chunkJson.id
                    };
                    this.messages.push(this.streamingTextMessage);
                    this.eventMessageIndexArray[index] = resultPart.text;
                    this.streamingTextMessage = null;
                  } 
                }
              }
            }
          }
        }
      },
      error: (err) => console.error('SSE error:', err),
      complete: () => {
        this.streamingTextMessage = null;
        this.sessionTab.reloadSession(this.sessionId);
        this.eventService.getTrace(this.sessionId)
            .pipe(catchError((error) => {
              if (error.status === 404) {
                return of(null);
              }
              return of([]);
            }))
            .subscribe(res => {this.traceData = res})
      },
    });
    this.userInput = '';
  }

  protected handleStarterQuestionSelected(questionText: string): void {
    this.userInput = questionText;
    const syntheticEvent = new Event('starterQuestionSelected'); // Create a generic event
    this.sendMessage(syntheticEvent);
    // sendMessage will set showStarterQuestions to false
  }


  private processPart(chunkJson: any, part: any, index: number) {
    if (part.text) {
      const newChunk = part.text;
      if (!this.streamingTextMessage) {
        this.streamingTextMessage = {
          role: 'bot',
          text: this.processThoughtText(newChunk),
          thought: part.thought ? true : false,
          eventId: chunkJson.id
        };

        if (chunkJson.groundingMetadata &&
            chunkJson.groundingMetadata.searchEntryPoint &&
            chunkJson.groundingMetadata.searchEntryPoint.renderedContent) {
          this.streamingTextMessage.renderedContent =
              chunkJson.groundingMetadata.searchEntryPoint.renderedContent;
        }

        this.messages.push(this.streamingTextMessage);
        if (!this.useSse) {
          this.storeEvents(part, chunkJson, index);
          this.eventMessageIndexArray[index] = newChunk;
          this.streamingTextMessage = null;
          return;
        }
      } else {
        if (newChunk == this.streamingTextMessage.text) {
          this.storeEvents(part, chunkJson, index);
          this.eventMessageIndexArray[index] = newChunk;
          this.streamingTextMessage = null;
          return;
        }
        this.streamingTextMessage.text += newChunk;
      }
    } else {
      this.storeEvents(part, chunkJson, index);
      this.storeMessage(part, chunkJson, index);
    }
  }

  async getUserMessageParts() {
    let parts: any = [{'text': `${this.userInput}`}];
    if (this.selectedFiles.length > 0) {
      for (const file of this.selectedFiles) {
        parts.push({
          inlineData: {
            data: await this.readFileAsBytes(file.file),
            mimeType: file.file.type,
          },
        });
      }
    }
    return parts;
  }

  readFileAsBytes(file: File): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        const base64Data = e.target.result.split(',')[1];
        resolve(base64Data);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  private updateRedirectUri(urlString: string, newRedirectUri: string): string {
    try {
      const url = new URL(urlString);
      const searchParams = url.searchParams;
      searchParams.set('redirect_uri', newRedirectUri);
      return url.toString();
    } catch (error) {
      console.warn('Failed to update redirect URI: ', error);
      return urlString;
    }
  }

  private storeMessage(part: any, e: any, index: number) {
    if (e.longRunningToolIds && e.longRunningToolIds.length > 0) {
      this.getAsyncFunctionsFromParts(e.longRunningToolIds, e.content.parts);
      const func = this.longRunningEvents[0];
      if (func.args.authConfig &&
          func.args.authConfig.exchangedAuthCredential &&
          func.args.authConfig.exchangedAuthCredential.oauth2) {
        const authUri =
            func.args.authConfig.exchangedAuthCredential.oauth2.authUri;
        const updatedAuthUri = this.updateRedirectUri(
            authUri,
            this.redirectUri,
        );
        this.openOAuthPopup(updatedAuthUri)
            .then((authResponseUrl) => {
              this.functionCallEventId = e.id;
              this.sendOAuthResponse(func, authResponseUrl, this.redirectUri);
            })
            .catch((error) => {
              console.error('OAuth Error:', error);
            });
      } else {
        this.functionCallEventId = e.id;
      }
    }
    if (e.actions && e.actions.artifactDelta) {
      for (const key in e.actions.artifactDelta) {
        if (e.actions.artifactDelta.hasOwnProperty(key)) {
          this.renderArtifact(key, e.actions.artifactDelta[key]);
        }
      }
    }
    if (part.text) {
      let message: any = {
        role: e.author === 'user' ? 'user' : 'bot',
        text: part.text,
        evalStatus: e.evalStatus,
        actualInvocationToolUses: e.actualInvocationToolUses,
        expectedInvocationToolUses: e.expectedInvocationToolUses
      };
      if (e.groundingMetadata && e.groundingMetadata.searchEntryPoint &&
          e.groundingMetadata.searchEntryPoint.renderedContent) {
        message.renderedContent =
            e.groundingMetadata.searchEntryPoint.renderedContent;
      }
      this.messages.push(message);
      this.eventMessageIndexArray[index] = part.text;
    } else if (part.functionCall) {
      this.messages.push({
        role: e.author === 'user' ? 'user' : 'bot',
        functionCall: part.functionCall,
        eventId: e.id,
        evalStatus: e.evalStatus,
        actualInvocationToolUses: e.actualInvocationToolUses,
        expectedInvocationToolUses: e.expectedInvocationToolUses,
      });
      this.eventMessageIndexArray[index] = part.functionCall;
    } else if (part.functionResponse) {
      this.messages.push({
        role: e.author === 'user' ? 'user' : 'bot',
        functionResponse: part.functionResponse,
        eventId: e.id,
        evalStatus: e.evalStatus,
        actualInvocationToolUses: e.actualInvocationToolUses,
        expectedInvocationToolUses: e.expectedInvocationToolUses,
      });

      this.eventMessageIndexArray[index] = part.functionResponse;
    } else if (part.executableCode) {
      this.messages.push({
        role: e.author === 'user' ? 'user' : 'bot',
        executableCode: part.executableCode,
        evalStatus: e.evalStatus,
        actualInvocationToolUses: e.actualInvocationToolUses,
        expectedInvocationToolUses: e.expectedInvocationToolUses,
      });
      this.eventMessageIndexArray[index] = part.executableCode;
    } else if (part.codeExecutionResult) {
      this.messages.push({
        role: e.author === 'user' ? 'user' : 'bot',
        codeExecutionResult: part.codeExecutionResult,
        evalStatus: e.evalStatus,
        actualInvocationToolUses: e.actualInvocationToolUses,
        expectedInvocationToolUses: e.expectedInvocationToolUses,
      });
      this.eventMessageIndexArray[index] = part.codeExecutionResult;
      if (e.actions && e.actions.artifact_delta) {
        for (const key in e.actions.artifact_delta) {
          if (e.actions.artifact_delta.hasOwnProperty(key)) {
            this.renderArtifact(key, e.actions.artifact_delta[key]);
          }
        }
      }
    }
  }

  private renderArtifact(artifactId: string, versionId: string) {
    this.messages.push({
      role: 'bot',
      inlineData: {
        data: '',
        mimeType: 'image/png',
      },
    });

    const currentIndex = this.messages.length - 1;

    this.artifactService
        .getArtifactVersion(
            this.userId,
            this.appName,
            this.sessionId,
            artifactId,
            versionId,
            )
        .subscribe((res) => {
          const mimeType = res.inlineData.mimeType;
          const fixedBase64Data = fixBase64String(res.inlineData.data);
          const base64Data = `data:${mimeType};base64,${fixedBase64Data}`;

          this.messages[currentIndex] = {
            role: 'bot',
            inlineData: {
              name: this.createDefaultArtifactName(mimeType),
              data: base64Data,
              mimeType: mimeType,
            },
          };

          this.artifacts = [
            ...this.artifacts,
            {
              id: artifactId,
              data: base64Data,
              mimeType,
              versionId,
            },
          ];
        });
  }

  private storeEvents(part: any, e: any, index: number) {
    let title = '';
    if (part.text) {
      title += 'text:' + part.text;
    } else if (part.functionCall) {
      title += 'functionCall:' + part.functionCall.name;
    } else if (part.functionResponse) {
      title += 'functionResponse:' + part.functionResponse.name;
    } else if (part.executableCode) {
      title += 'executableCode:' + part.executableCode.code.slice(0, 10);
    } else if (part.codeExecutionResult) {
      title += 'codeExecutionResult:' + part.codeExecutionResult.outcome;
    }
    e.title = title;

    this.eventData.set(e.id, e);
    this.eventData = new Map(this.eventData);
  }

  private sendOAuthResponse(
      func: any,
      authResponseUrl: string,
      redirectUri: string,
  ) {
    this.longRunningEvents.pop();
    const authResponse: AgentRunRequest = {
      appName: this.appName,
      userId: this.userId,
      sessionId: this.sessionId,
      newMessage: {
        'role': 'user',
        'parts': [],
      },
    };

    var authConfig = structuredClone(func.args.authConfig);
    authConfig.exchangedAuthCredential.oauth2.authResponseUri = authResponseUrl;
    authConfig.exchangedAuthCredential.oauth2.redirectUri = redirectUri;

    authResponse.functionCallEventId = this.functionCallEventId;
    authResponse.newMessage.parts.push({
      'function_response': {
        id: func.id,
        name: func.name,
        response: authConfig,
      },
    });
    this.agentService.run(authResponse).subscribe((res) => {
      let index = this.eventMessageIndexArray.length - 1;
      for (const e of res) {
        if (e.content) {
          for (let part of e.content.parts) {
            index += 1;
            this.processPart(e, part, index);
          }
        }
      }
    });
  }

  openDialog(): void {
    const dialogRef = this.dialog.open(PendingEventDialogComponent, {
      width: '600px',
      data: {
        event: this.longRunningEvents[0],
        appName: this.appName,
        userId: this.userId,
        sessionId: this.sessionId,
        functionCallEventId: this.functionCallEventId,
      },
    });

    dialogRef.afterClosed().subscribe((t) => {
      if (t) {
        this.removeFinishedLongRunningEvents(t.events);
        this.messages.push({
          role: 'bot',
          text: t.text,
        });
      }
    });
  }

  removeFinishedLongRunningEvents(finishedEvents: any[]) {
    const idsToExclude = new Set(finishedEvents.map((obj: any) => obj.id));
    this.longRunningEvents =
        this.longRunningEvents.filter(obj => !idsToExclude.has(obj.id));
  }

  clickEvent(i: number) {
    const key = this.messages[i].eventId;
    this.sideDrawer.open();
    this.showSidePanel = true;
    this.selectedEvent = this.eventData.get(key);
    this.selectedEventIndex = this.getIndexOfKeyInMap(key);

    this.eventService.getEventTrace(this.selectedEvent.id).subscribe((res) => {
      this.llmRequest = JSON.parse(res[this.llmRequestKey]);
      this.llmResponse = JSON.parse(res[this.llmResponseKey]);
    });

    this.eventService
        .getEvent(
            this.userId,
            this.appName,
            this.sessionId,
            this.selectedEvent.id,
            )
        .subscribe(async (res) => {
          if (!res.dotSrc) {
            this.renderedEventGraph = undefined;
            return;
          }
          const graphSrc = res.dotSrc;
          const viz = await instance();
          const svg = viz.renderString(graphSrc, {
            format: 'svg',
            engine: 'dot',
          });
          this.rawSvgString = svg;
          this.renderedEventGraph = this.sanitizer.bypassSecurityTrustHtml(svg);
        });
  }

  userMessagesLength(i: number) {
    return this.messages.slice(0, i).filter((m) => m.role == 'user').length;
  }

  ngOnDestroy(): void {
    this.webSocketService.closeConnection();
  }

  toggleAudioRecording() {
    this.isAudioRecording ? this.stopAudioRecording() :
                            this.startAudioRecording();
  }

  startAudioRecording() {
    if (this.sessionHasUsedBidi.has(this.sessionId)) {
      this.openSnackBar(BIDI_STREAMING_RESTART_WARNING, 'OK')
      return;
    }

    this.isAudioRecording = true;
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    this.webSocketService.connect(
        `${protocol}://${URLUtil.getWSServerUrl()}/run_live?app_name=${
            this.appName}&user_id=${this.userId}&session_id=${this.sessionId}`,
    );
    this.audioService.startRecording();
    this.messages.push({role: 'user', text: 'Speaking...'});
    this.messages.push({role: 'bot', text: 'Speaking...'});
    this.sessionHasUsedBidi.add(this.sessionId);
  }

  stopAudioRecording() {
    this.audioService.stopRecording();
    this.webSocketService.closeConnection();
    this.isAudioRecording = false;
  }

  toggleVideoRecording() {
    this.isVideoRecording ? this.stopVideoRecording() :
                            this.startVideoRecording();
  }

  startVideoRecording() {
    if (this.sessionHasUsedBidi.has(this.sessionId)) {
      this.openSnackBar(BIDI_STREAMING_RESTART_WARNING, 'OK')
      return;
    }

    this.isVideoRecording = true;
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    this.webSocketService.connect(
        `${protocol}://${URLUtil.getWSServerUrl()}/run_live?app_name=${
            this.appName}&user_id=${this.userId}&session_id=${this.sessionId}`,
    );
    this.videoService.startRecording(this.videoContainer);
    this.audioService.startRecording();
    this.messages.push({role: 'user', text: 'Speaking...'});
    this.sessionHasUsedBidi.add(this.sessionId);
  }

  stopVideoRecording() {
    this.audioService.stopRecording();
    this.videoService.stopRecording(this.videoContainer);
    this.webSocketService.closeConnection();
    this.isVideoRecording = false;
  }

  private getAsyncFunctionsFromParts(pendingIds: any[], parts: any[]) {
    for (const part of parts) {
      if (part.functionCall && pendingIds.includes(part.functionCall.id)) {
        this.longRunningEvents.push(part.functionCall);
      }
    }
  }

  private openOAuthPopup(url: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const popup = window.open(url, 'oauthPopup', 'width=600,height=700');
      if (!popup) {
        reject('Popup blocked!');
        return;
      }
      window.addEventListener(
          'message',
          (event) => {
            if (event.origin !== window.location.origin) {
              return;
            }
            const {authResponseUrl} = event.data;
            if (authResponseUrl) {
              resolve(authResponseUrl);
            } else {
              reject('OAuth failed');
            }
          },
          {once: true},
      );
    });
  }

  toggleSidePanel() {
    if (this.showSidePanel) {
      this.sideDrawer.close();
    } else {
      this.sideDrawer.open();
    }
    this.showSidePanel = !this.showSidePanel;
  }

  protected handleShouldShowEvalTab(shouldShow: boolean) {
    this.shouldShowEvalTab.set(shouldShow);
  }

  protected updateWithSelectedSession(session: Session) {
    if (!session || !session.id || !session.events || !session.state) {
      this.showStarterQuestions = true; // Or false, depending on desired behavior for invalid session
      return;
    }

    this.sessionId = session.id;
    this.currentSessionState = session.state;

    this.eventData.clear();
    this.eventMessageIndexArray = [];
    this.messages = []; // Clear messages before populating
    this.artifacts = [];
    let index = 0;

    session.events.forEach((event: any) => {
      event.content?.parts?.forEach((part: any) => {
        this.storeMessage(part, event, index);
        index += 1;
        if (event.author && event.author !== 'user') {
          this.storeEvents(part, event, index);
        }
      });
    });

    // Show starter questions only if the loaded session is empty
    this.showStarterQuestions = this.messages.length === 0;

    this.eventService.getTrace(this.sessionId).subscribe(res => {
      this.traceData = res;
    })
  }

  protected updateSessionState(session: Session) {
    this.currentSessionState = session.state;
  }

  onNewSessionClick() {
    this.selectApp(HARDCODED_AGENT_NAME); // This will call createSession, which clears messages and sets showStarterQuestions = true.
    if (this.evalTab && !!this.evalTab.showEvalHistory) {
      this.evalTab.toggleEvalHistoryButton();
    }
  }

  onFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      for (let i = 0; i < input.files.length; i++) {
        const file = input.files[i];
        const url = URL.createObjectURL(file);
        this.selectedFiles.push({file, url});
      }
    }
    input.value = '';
  }

  removeFile(index: number) {
    URL.revokeObjectURL(this.selectedFiles[index].url);
    this.selectedFiles.splice(index, 1);
  }

  toggleSse() {
    this.useSse = !this.useSse;
  }

  selectEvent(key: string) {
    this.selectedEvent = this.eventData.get(key);
    this.selectedEventIndex = this.getIndexOfKeyInMap(key);
    this.eventService.getEventTrace(this.selectedEvent.id).subscribe((res) => {
      this.llmRequest = JSON.parse(res[this.llmRequestKey]);
      this.llmResponse = JSON.parse(res[this.llmResponseKey]);
    });
    this.eventService
        .getEvent(
            this.userId,
            this.appName,
            this.sessionId,
            this.selectedEvent.id,
            )
        .subscribe(async (res) => {
          if (!res.dotSrc) {
            this.renderedEventGraph = undefined;
            return;
          }
          const graphSrc = res.dotSrc;
          const viz = await instance();
          const svg = viz.renderString(graphSrc, {
            format: 'svg',
            engine: 'dot',
          });
          this.rawSvgString = svg;
          this.renderedEventGraph = this.sanitizer.bypassSecurityTrustHtml(svg);
        });
  }

  protected deleteSession(session: string) {
    const dialogData: DeleteSessionDialogData = {
      title: 'Confirm delete',
      message:
          `Are you sure you want to delete this session ${this.sessionId}?`,
      confirmButtonText: 'Delete',
      cancelButtonText: 'Cancel',
    };

    const dialogRef = this.dialog.open(DeleteSessionDialogComponent, {
      width: '600px',
      data: dialogData,
    });

    dialogRef.afterClosed().subscribe((result) => {
      if (result) {
        this.sessionService.deleteSession(this.userId, this.appName, session)
            .subscribe((res) => {
              const nextSession = this.sessionTab.refreshSession(session);
              if (nextSession) {
                // getSession will trigger updateWithSelectedSession, which handles showStarterQuestions
                this.sessionTab.getSession(nextSession.id);
              } else {
                // No next session, so create a new one for the hardcoded agent
                this.selectApp(HARDCODED_AGENT_NAME);
              }
            });
      }
    });
  }

  handlePageEvent(event: any) {
    if (event.pageIndex >= 0) {
      const key = this.getKeyAtIndexInMap(event.pageIndex);
      if (key) {
        this.selectEvent(key);
      }
    }
  }

  closeSelectedEvent() {
    this.selectedEvent = undefined;
    this.selectedEventIndex = undefined;
  }

  private getIndexOfKeyInMap(key: string): number|undefined {
    let index = 0;
    const mapOrderPreservingSort = (a: any, b: any): number => 0;
    const sortedKeys = Array.from(this.eventData.keys())
                           .sort(mapOrderPreservingSort);
    for (const k of sortedKeys) {
      if (k === key) {
        return index;
      }
      index++;
    }
    return undefined;
  }

  private getKeyAtIndexInMap(index: number): string|undefined {
    const mapOrderPreservingSort = (a: any, b: any): number => 0;
    const sortedKeys = Array.from(this.eventData.keys())
                           .sort(mapOrderPreservingSort);
    if (index >= 0 && index < sortedKeys.length) {
      return sortedKeys[index];
    }
    return undefined;
  }

  openSnackBar(message: string, action: string) {
    this._snackBar.open(message, action);
  }

  private processThoughtText(text: string) {
    return text.replace('/*PLANNING*/', '').replace('/*ACTION*/', '');
  }

  openLink(url: string) {
    window.open(url, '_blank');
  }

  renderGooglerSearch(content: string) {
    return this.sanitizer.bypassSecurityTrustHtml(content);
  }

  openViewImageDialog(imageData: string|null) {
    const dialogRef = this.dialog.open(ViewImageDialogComponent, {
      data: {
        imageData,
      },
    });
  }

  private createDefaultArtifactName(mimeType: string): string {
    if (!mimeType || !mimeType.includes('/')) {
      return '';
    }
    return mimeType.replace('/', '.');
  }
}