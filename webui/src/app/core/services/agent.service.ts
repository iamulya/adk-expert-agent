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

import {HttpClient} from '@angular/common/http';
import {Injectable, NgZone} from '@angular/core';
import {BehaviorSubject, Observable, of} from 'rxjs';
import {URLUtil} from '../../../utils/url-util';
import {AgentRunRequest} from '../models/AgentRunRequest';

@Injectable({
  providedIn: 'root',
})
export class AgentService {
  apiServerDomain = URLUtil.getApiServerBaseUrl();
  private _currentApp = new BehaviorSubject<string>('');
  currentApp = this._currentApp.asObservable();
  private isLoading = new BehaviorSubject<boolean>(false);

  constructor(
    private http: HttpClient,
    private zone: NgZone,
  ) {}

  getApp(): Observable<string> {
    return this.currentApp;
  }

  setApp(name: string) {
    this._currentApp.next(name);
  }

  getLoadingState(): BehaviorSubject<boolean> {
    return this.isLoading;
  }

  run(req: AgentRunRequest) {
    const headers = {
      'Content-type': 'application/json',
    };
    const options = {
      headers: headers,
    };

    const url = this.apiServerDomain + `/run`;
    return this.http.post<any>(url, req, options);
  }

  runSse(req: AgentRunRequest): Observable<string> {
    const url = `${this.apiServerDomain}/run_sse`;
    this.isLoading.next(true);

    return new Observable<string>((observer) => {
      const self = this;
      const controller = new AbortController();

      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(req),
        signal: controller.signal, // For cancellation
      })
        .then(async (response) => {
          if (!response.body) {
            throw new Error('Response body is null');
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              // Process any remaining data in the buffer before completing
              if (buffer.startsWith('data: ')) {
                const data = buffer.substring(6);
                self.zone.run(() => observer.next(data));
              }
              observer.complete();
              return;
            }

            buffer += decoder.decode(value, { stream: true });

            // Process complete messages from the buffer
            let endOfMessageIndex;
            while ((endOfMessageIndex = buffer.indexOf('\n\n')) >= 0) {
              const message = buffer.slice(0, endOfMessageIndex);
              buffer = buffer.slice(endOfMessageIndex + 2); // Skip the '\n\n'

              if (message.startsWith('data: ')) {
                const data = message.substring(6); // Remove 'data: ' prefix
                if (data.trim()) {
                  self.zone.run(() => observer.next(data));
                }
              }
            }
          }
        })
        .catch((err) => {
          if (err.name !== 'AbortError') {
             self.zone.run(() => observer.error(err));
          }
        });
      
      // Return a teardown function to abort the fetch request
      return () => {
        controller.abort();
      };
    });
  }


  listApps(): Observable<string[]> {
    if (this.apiServerDomain != undefined) {
      const url = this.apiServerDomain + `/list-apps?relative_path=./`;
      return this.http.get<string[]>(url);
    }
    return new Observable<[]>();
  }
}